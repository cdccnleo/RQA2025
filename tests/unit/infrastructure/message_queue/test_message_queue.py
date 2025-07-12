#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
消息队列模块单元测试
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from queue import Queue, Empty, Full
from typing import Dict, Any

# 模拟消息队列模块
class MessageQueue:
    """消息队列实现"""
    
    def __init__(self, maxsize: int = 1000):
        self.queue = Queue(maxsize=maxsize)
        self.consumers = []
        self.running = False
        self._lock = threading.Lock()
        
    def put(self, message: Dict[str, Any], timeout: float = 1.0) -> bool:
        """发送消息到队列"""
        try:
            self.queue.put(message, timeout=timeout)
            return True
        except Full:
            return False
            
    def get(self, timeout: float = 1.0) -> Dict[str, Any]:
        """从队列获取消息"""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None
            
    def register_consumer(self, consumer_func):
        """注册消费者"""
        with self._lock:
            self.consumers.append(consumer_func)
            
    def start_consuming(self):
        """开始消费消息"""
        self.running = True
        thread = threading.Thread(target=self._consume_worker, daemon=True)
        thread.start()
        
    def stop_consuming(self):
        """停止消费消息"""
        self.running = False
        
    def _consume_worker(self):
        """消费工作线程"""
        while self.running:
            message = self.get(timeout=0.1)
            if message:
                for consumer in self.consumers:
                    try:
                        consumer(message)
                    except Exception as e:
                        # 记录错误但不中断处理
                        print(f"Consumer error: {e}")

class TestMessageQueue:
    """消息队列测试类"""
    
    @pytest.fixture
    def message_queue(self):
        """创建消息队列实例"""
        return MessageQueue(maxsize=10)
        
    def test_queue_initialization(self, message_queue):
        """测试队列初始化"""
        assert message_queue.queue.maxsize == 10
        assert len(message_queue.consumers) == 0
        assert not message_queue.running
        
    def test_put_message_success(self, message_queue):
        """测试成功发送消息"""
        message = {"type": "test", "data": "hello"}
        result = message_queue.put(message)
        assert result is True
        
        # 验证消息在队列中
        assert not message_queue.queue.empty()
        
    def test_put_message_queue_full(self, message_queue):
        """测试队列满时发送消息"""
        # 填满队列
        for i in range(10):
            message_queue.put({"id": i})
            
        # 尝试发送第11条消息
        result = message_queue.put({"id": 11})
        assert result is False
        
    def test_get_message_success(self, message_queue):
        """测试成功获取消息"""
        message = {"type": "test", "data": "hello"}
        message_queue.put(message)
        
        received = message_queue.get()
        assert received == message
        
    def test_get_message_empty_queue(self, message_queue):
        """测试空队列获取消息"""
        received = message_queue.get(timeout=0.1)
        assert received is None
        
    def test_register_consumer(self, message_queue):
        """测试注册消费者"""
        consumer = Mock()
        message_queue.register_consumer(consumer)
        
        assert len(message_queue.consumers) == 1
        assert consumer in message_queue.consumers
        
    def test_multiple_consumers(self, message_queue):
        """测试多个消费者"""
        consumer1 = Mock()
        consumer2 = Mock()
        
        message_queue.register_consumer(consumer1)
        message_queue.register_consumer(consumer2)
        
        assert len(message_queue.consumers) == 2
        assert consumer1 in message_queue.consumers
        assert consumer2 in message_queue.consumers
        
    def test_consumer_processing(self, message_queue):
        """测试消费者处理消息"""
        processed_messages = []
        
        def consumer(message):
            processed_messages.append(message)
            
        message_queue.register_consumer(consumer)
        message_queue.start_consuming()
        
        # 发送消息
        test_message = {"type": "test", "data": "hello"}
        message_queue.put(test_message)
        
        # 等待处理
        time.sleep(0.1)
        
        message_queue.stop_consuming()
        
        assert len(processed_messages) == 1
        assert processed_messages[0] == test_message
        
    def test_consumer_error_handling(self, message_queue):
        """测试消费者错误处理"""
        error_consumer = Mock(side_effect=Exception("Test error"))
        success_consumer = Mock()
        
        message_queue.register_consumer(error_consumer)
        message_queue.register_consumer(success_consumer)
        message_queue.start_consuming()
        
        # 发送消息
        test_message = {"type": "test", "data": "hello"}
        message_queue.put(test_message)
        
        # 等待处理
        time.sleep(0.1)
        
        message_queue.stop_consuming()
        
        # 验证两个消费者都被调用
        error_consumer.assert_called_once_with(test_message)
        success_consumer.assert_called_once_with(test_message)
        
    def test_thread_safety(self, message_queue):
        """测试线程安全性"""
        results = []
        
        def producer():
            for i in range(5):
                message_queue.put({"id": i})
                time.sleep(0.01)
                
        def consumer(message):
            results.append(message["id"])
            
        message_queue.register_consumer(consumer)
        message_queue.start_consuming()
        
        # 启动生产者线程
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        producer_thread.join()
        
        # 等待消费完成
        time.sleep(0.1)
        message_queue.stop_consuming()
        
        # 验证所有消息都被处理
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}
        
    def test_queue_size_monitoring(self, message_queue):
        """测试队列大小监控"""
        assert message_queue.queue.qsize() == 0
        
        # 添加消息
        message_queue.put({"id": 1})
        assert message_queue.queue.qsize() == 1
        
        # 获取消息
        message_queue.get()
        assert message_queue.queue.qsize() == 0
        
    def test_timeout_handling(self, message_queue):
        """测试超时处理"""
        # 测试获取超时
        start_time = time.time()
        result = message_queue.get(timeout=0.1)
        end_time = time.time()
        
        assert result is None
        assert end_time - start_time >= 0.1
        
        # 测试发送超时
        # 填满队列
        for i in range(10):
            message_queue.put({"id": i})
            
        start_time = time.time()
        result = message_queue.put({"id": 11}, timeout=0.1)
        end_time = time.time()
        
        assert result is False
        assert end_time - start_time >= 0.1

class TestMessageQueueIntegration:
    """消息队列集成测试"""
    
    def test_end_to_end_message_flow(self):
        """测试端到端消息流"""
        queue = MessageQueue(maxsize=5)
        received_messages = []
        
        def message_handler(message):
            received_messages.append(message)
            
        queue.register_consumer(message_handler)
        queue.start_consuming()
        
        # 发送多条消息
        messages = [
            {"type": "order", "data": "buy 100 shares"},
            {"type": "trade", "data": "sell 50 shares"},
            {"type": "alert", "data": "price threshold exceeded"}
        ]
        
        for message in messages:
            queue.put(message)
            
        # 等待处理
        time.sleep(0.2)
        queue.stop_consuming()
        
        # 验证所有消息都被处理
        assert len(received_messages) == 3
        assert received_messages == messages
        
    def test_concurrent_producers_consumers(self):
        """测试并发生产者和消费者"""
        queue = MessageQueue(maxsize=20)
        processed_count = 0
        lock = threading.Lock()
        
        def consumer(message):
            nonlocal processed_count
            with lock:
                processed_count += 1
                
        queue.register_consumer(consumer)
        queue.start_consuming()
        
        def producer(producer_id):
            for i in range(10):
                queue.put({"producer": producer_id, "message": i})
                time.sleep(0.01)
                
        # 启动多个生产者
        producers = []
        for i in range(3):
            thread = threading.Thread(target=producer, args=(i,))
            producers.append(thread)
            thread.start()
            
        # 等待所有生产者完成
        for thread in producers:
            thread.join()
            
        # 等待消费完成
        time.sleep(0.5)
        queue.stop_consuming()
        
        # 验证所有消息都被处理
        assert processed_count == 30  # 3个生产者 * 10条消息
        
    def test_error_recovery(self):
        """测试错误恢复"""
        queue = MessageQueue(maxsize=10)
        processed_messages = []
        error_count = 0
        
        def error_consumer(message):
            nonlocal error_count
            if message.get("should_fail"):
                error_count += 1
                raise Exception("Simulated error")
            else:
                processed_messages.append(message)
                
        queue.register_consumer(error_consumer)
        queue.start_consuming()
        
        # 发送混合消息
        messages = [
            {"id": 1, "should_fail": False},
            {"id": 2, "should_fail": True},
            {"id": 3, "should_fail": False},
            {"id": 4, "should_fail": True},
            {"id": 5, "should_fail": False}
        ]
        
        for message in messages:
            queue.put(message)
            
        # 等待处理
        time.sleep(0.2)
        queue.stop_consuming()
        
        # 验证错误处理
        assert error_count == 2
        assert len(processed_messages) == 3
        assert processed_messages == [
            {"id": 1, "should_fail": False},
            {"id": 3, "should_fail": False},
            {"id": 5, "should_fail": False}
        ]

if __name__ == "__main__":
    pytest.main([__file__]) 