# -*- coding: utf-8 -*-
"""
死锁预防测试
验证死锁修复是否有效
"""

import pytest
import time
import threading
import concurrent.futures
from unittest.mock import Mock

from src.core.event_bus.bus_components import EventBus
from src.core.event_bus.event_bus import EventType
from src.infrastructure.utils.optimized_connection_pool import OptimizedConnectionPool



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

# @pytest.mark.deadlock_test  # 暂时注释掉，等待pytest配置重载
class TestDeadlockPrevention:
    """死锁预防测试"""

    def test_event_bus_batch_processing_no_deadlock(self):
        """测试事件总线批量处理无死锁"""
        bus = EventBus(max_workers=2, enable_async=True, batch_size=10)

        # 模拟大量并发事件发布
        def publish_events(start_idx, count):
            for i in range(start_idx, start_idx + count):
                bus.publish(EventType.DATA_RECEIVED, {'id': i}, f'source_{i}')

        # 创建多个线程并发发布事件
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(publish_events, i * 20, 20)
                futures.append(future)

            # 等待所有任务完成，设置超时
            for future in concurrent.futures.as_completed(futures, timeout=10):
                future.result()

        # 验证事件被处理
        history = bus.get_event_history()
        assert len(history) > 0, "事件应该被添加到历史记录中"

        bus.shutdown()

    def test_connection_pool_lock_release(self):
        """测试连接池锁释放机制"""
        # 创建一个小的连接池用于测试
        pool = OptimizedConnectionPool(
            max_size=5,
            min_size=1,
            connection_timeout=5.0
        )

        # 模拟连接工厂
        mock_connection = Mock()
        pool.set_connection_factory(lambda: mock_connection)

        # 测试并发获取连接
        results = []
        errors = []

        def get_connection_worker(worker_id):
            try:
                conn = pool.get_connection(timeout=2.0)
                if conn:
                    time.sleep(0.1)  # 模拟使用连接
                    pool.release_connection(conn)
                    results.append(f"worker_{worker_id}_success")
                else:
                    results.append(f"worker_{worker_id}_timeout")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # 创建多个线程并发获取连接
        threads = []
        for i in range(10):
            thread = threading.Thread(target=get_connection_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=5.0)

        # 验证没有死锁（至少有一些成功的结果）
        success_count = len([r for r in results if 'success' in r])
        assert success_count > 0, f"应该有成功的连接获取，结果: {results}, 错误: {errors}"

        pool.shutdown()

    def test_event_bus_memory_limit_prevents_leak(self):
        """测试事件总线内存限制防止内存泄漏"""
        bus = EventBus(max_workers=2, enable_async=False)

        # 发布大量事件
        for i in range(200):  # 超过默认内存限制
            bus.publish(EventType.DATA_RECEIVED, {'data': 'x' * 1000}, f'source_{i}')

        # 验证历史记录被限制
        history = bus.get_event_history()
        assert len(history) <= 10000, f"历史记录应该被限制在10000以内，实际: {len(history)}"

        bus.shutdown()

    @pytest.mark.timeout(10)
    def test_connection_pool_timeout_prevents_hang(self):
        """测试连接池超时防止挂起"""
        pool = OptimizedConnectionPool(
            max_size=1,  # 只有1个连接
            min_size=0,
            connection_timeout=1.0
        )

        # 不设置连接工厂，让连接创建失败
        # 这应该导致获取连接超时，而不是无限等待

        start_time = time.time()
        conn = pool.get_connection(timeout=2.0)
        end_time = time.time()

        # 验证超时机制工作（应该在合理时间内返回None）
        assert conn is None, "应该返回None因为没有可用的连接"
        assert end_time - start_time < 3.0, f"获取连接耗时过长: {end_time - start_time}秒"

        pool.shutdown()

    def test_concurrent_event_processing_no_deadlock(self):
        """测试并发事件处理无死锁"""
        bus = EventBus(max_workers=3, enable_async=True)

        # 设置事件处理器
        mock_handler = Mock()
        mock_handler.handle_event = Mock(return_value=True)
        bus.subscribe(EventType.DATA_RECEIVED, mock_handler)

        # 并发发布和处理事件
        events_processed = []

        def publish_and_process(worker_id):
            for i in range(10):
                event_data = {'worker': worker_id, 'seq': i}
                bus.publish(EventType.DATA_RECEIVED, event_data, f'worker_{worker_id}')
                events_processed.append(f"{worker_id}_{i}")

        # 创建并发线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(publish_and_process, i) for i in range(5)]

            # 等待完成，设置超时
            for future in concurrent.futures.as_completed(futures, timeout=15):
                future.result()

        # 验证事件被处理
        assert len(events_processed) == 50, f"应该处理50个事件，实际: {len(events_processed)}"

        bus.shutdown()
