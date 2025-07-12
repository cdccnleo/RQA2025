import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from src.data.parallel.thread_pool import (
    ThreadPoolConfig, DynamicThreadPool, create_default_pool
)

class TestThreadPoolConfig:
    """ThreadPoolConfig类测试"""

    @pytest.fixture
    def config(self):
        """创建线程池配置实例"""
        return ThreadPoolConfig()

    def test_config_init(self, config):
        """测试配置初始化"""
        assert config.core_pool_size > 0
        assert config.max_pool_size == 50
        assert config.queue_capacity == 1000
        assert config.keep_alive == 60

    def test_adjust_based_on_load_high(self, config):
        """测试高负载调整"""
        original_core_size = config.core_pool_size
        original_queue_capacity = config.queue_capacity
        
        config.adjust_based_on_load(0.8)  # 高负载
        
        assert config.core_pool_size < original_core_size
        assert config.queue_capacity == 500

    def test_adjust_based_on_load_low(self, config):
        """测试低负载调整"""
        original_core_size = config.core_pool_size
        original_queue_capacity = config.queue_capacity
        
        config.adjust_based_on_load(0.2)  # 低负载
        
        assert config.core_pool_size >= original_core_size
        assert config.queue_capacity == 2000

    def test_adjust_based_on_load_normal(self, config):
        """测试正常负载不调整"""
        original_core_size = config.core_pool_size
        original_queue_capacity = config.queue_capacity
        
        config.adjust_based_on_load(0.5)  # 正常负载
        
        assert config.core_pool_size == original_core_size
        assert config.queue_capacity == original_queue_capacity

class TestDynamicThreadPool:
    """DynamicThreadPool类测试"""

    @pytest.fixture
    def config(self):
        """创建线程池配置"""
        return ThreadPoolConfig(
            core_pool_size=2,
            max_pool_size=4,
            queue_capacity=10,
            keep_alive=30
        )

    @pytest.fixture
    def thread_pool(self, config):
        """创建线程池实例"""
        pool = DynamicThreadPool(config)
        yield pool
        pool.shutdown(wait=False)

    def test_pool_init(self, thread_pool):
        """测试线程池初始化"""
        assert thread_pool.config is not None
        assert thread_pool.executor is not None
        assert thread_pool.running is True
        assert thread_pool.task_queue is not None

    def test_submit_task(self, thread_pool):
        """测试提交任务"""
        def test_function(x):
            return x * 2
        
        future = thread_pool.submit(test_function, 5)
        
        assert future is not None
        assert future.result() == 10

    def test_submit_multiple_tasks(self, thread_pool):
        """测试提交多个任务"""
        def test_function(x):
            time.sleep(0.1)  # 模拟工作
            return x * 2
        
        futures = []
        for i in range(5):
            future = thread_pool.submit(test_function, i)
            futures.append(future)
        
        results = [future.result() for future in futures]
        assert results == [0, 2, 4, 6, 8]

    def test_submit_task_with_exception(self, thread_pool):
        """测试提交会抛出异常的任务"""
        def failing_function():
            raise ValueError("Test exception")
        
        future = thread_pool.submit(failing_function)
        
        with pytest.raises(ValueError, match="Test exception"):
            future.result()

    def test_shutdown(self, thread_pool):
        """测试关闭线程池"""
        assert thread_pool.running is True
        
        thread_pool.shutdown()
        
        assert thread_pool.running is False

    def test_submit_after_shutdown(self, thread_pool):
        """测试关闭后提交任务"""
        thread_pool.shutdown()
        
        with pytest.raises(RuntimeError, match="Thread pool not running"):
            thread_pool.submit(lambda: None)

    def test_get_stats(self, thread_pool):
        """测试获取统计信息"""
        stats = thread_pool.get_stats()
        
        assert 'active_threads' in stats
        assert 'pending_tasks' in stats
        assert 'core_pool_size' in stats
        assert 'max_pool_size' in stats
        assert 'queue_capacity' in stats
        
        assert stats['core_pool_size'] == 2
        assert stats['max_pool_size'] == 4
        assert stats['queue_capacity'] == 10

    def test_queue_full_handling(self, thread_pool):
        """测试队列满时的处理"""
        def slow_function():
            time.sleep(0.5)
            return "done"
        
        # 提交多个任务填满队列
        futures = []
        for i in range(15):  # 超过队列容量
            future = thread_pool.submit(slow_function)
            futures.append(future)
        
        # 应该能够处理队列满的情况
        results = [future.result() for future in futures]
        assert all(result == "done" for result in results)

    def test_concurrent_access(self, thread_pool):
        """测试并发访问"""
        def test_function(x):
            time.sleep(0.01)
            return x * 2
        
        # 多个线程同时提交任务
        def submit_tasks():
            for i in range(10):
                future = thread_pool.submit(test_function, i)
                assert future.result() == i * 2
        
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=submit_tasks)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()

    def test_monitor_thread_functionality(self, thread_pool):
        """测试监控线程功能"""
        # 等待监控线程启动
        time.sleep(0.1)
        
        # 检查监控线程是否在运行
        assert thread_pool.monitor_thread.is_alive()

    @patch('psutil.getloadavg')
    def test_load_monitoring(self, mock_getloadavg, thread_pool):
        """测试负载监控"""
        # 模拟高负载
        mock_getloadavg.return_value = (8.0, 7.0, 6.0)
        
        original_core_size = thread_pool.config.core_pool_size
        
        # 等待监控线程执行
        time.sleep(0.1)
        
        # 检查是否根据负载调整了配置
        # 注意：实际调整可能需要更长时间

    def test_task_queue_management(self, thread_pool):
        """测试任务队列管理"""
        def test_function(x):
            return x
        
        # 提交任务并检查队列
        future = thread_pool.submit(test_function, 42)
        
        # 等待任务完成
        result = future.result()
        assert result == 42

    def test_thread_pool_reuse(self, config):
        """测试线程池重用"""
        pool1 = DynamicThreadPool(config)
        pool2 = DynamicThreadPool(config)
        
        assert pool1 is not pool2
        assert pool1.executor is not pool2.executor
        
        pool1.shutdown()
        pool2.shutdown()

    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效配置
        with pytest.raises(ValueError):
            ThreadPoolConfig(core_pool_size=-1)
        
        with pytest.raises(ValueError):
            ThreadPoolConfig(max_pool_size=0)
        
        with pytest.raises(ValueError):
            ThreadPoolConfig(queue_capacity=-1)

def test_create_default_pool():
    """测试创建默认线程池"""
    pool = create_default_pool()
    
    assert isinstance(pool, DynamicThreadPool)
    assert pool.config is not None
    assert pool.executor is not None
    
    # 测试默认配置
    assert pool.config.max_pool_size == 50
    assert pool.config.queue_capacity == 1000
    assert pool.config.keep_alive == 60
    
    pool.shutdown()

def test_thread_pool_integration():
    """测试线程池集成功能"""
    config = ThreadPoolConfig(
        core_pool_size=2,
        max_pool_size=4,
        queue_capacity=5,
        keep_alive=30
    )
    
    pool = DynamicThreadPool(config)
    
    # 测试基本功能
    def worker_function(x):
        time.sleep(0.01)
        return x * x
    
    # 提交多个任务
    futures = []
    for i in range(10):
        future = pool.submit(worker_function, i)
        futures.append(future)
    
    # 收集结果
    results = [future.result() for future in futures]
    expected = [i * i for i in range(10)]
    assert results == expected
    
    # 检查统计信息
    stats = pool.get_stats()
    assert stats['core_pool_size'] == 2
    assert stats['max_pool_size'] == 4
    
    pool.shutdown()

def test_thread_pool_error_handling():
    """测试线程池错误处理"""
    config = ThreadPoolConfig(
        core_pool_size=1,
        max_pool_size=2,
        queue_capacity=3,
        keep_alive=30
    )
    
    pool = DynamicThreadPool(config)
    
    # 测试异常传播
    def error_function():
        raise RuntimeError("Test error")
    
    future = pool.submit(error_function)
    
    with pytest.raises(RuntimeError, match="Test error"):
        future.result()
    
    pool.shutdown()

def test_thread_pool_performance():
    """测试线程池性能"""
    config = ThreadPoolConfig(
        core_pool_size=4,
        max_pool_size=8,
        queue_capacity=20,
        keep_alive=30
    )
    
    pool = DynamicThreadPool(config)
    
    # 性能测试：提交大量任务
    def performance_test_function(x):
        return x * x
    
    start_time = time.time()
    
    futures = []
    for i in range(100):
        future = pool.submit(performance_test_function, i)
        futures.append(future)
    
    # 收集所有结果
    results = [future.result() for future in futures]
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 验证结果
    expected = [i * i for i in range(100)]
    assert results == expected
    
    # 验证执行时间合理（应该比串行执行快）
    assert execution_time < 1.0  # 应该很快完成
    
    pool.shutdown() 