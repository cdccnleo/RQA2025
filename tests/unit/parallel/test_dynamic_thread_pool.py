import unittest
import time
import os
from unittest.mock import patch
from src.data.parallel.thread_pool import (
    ThreadPoolConfig,
    DynamicThreadPool,
    create_default_pool
)

def mock_task(x):
    """模拟任务函数"""
    time.sleep(0.1)
    return x * x

class TestThreadPoolConfig(unittest.TestCase):
    """线程池配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = ThreadPoolConfig()
        self.assertEqual(config.core_pool_size, os.cpu_count() or 2)
        self.assertEqual(config.max_pool_size, 50)

    def test_load_adjustment(self):
        """测试负载调整"""
        config = ThreadPoolConfig()

        # 模拟高负载
        config.adjust_based_on_load(0.8)
        self.assertLess(config.core_pool_size, os.cpu_count() or 2)
        self.assertEqual(config.queue_capacity, 500)

        # 模拟低负载
        config.adjust_based_on_load(0.2)
        self.assertGreater(config.core_pool_size, os.cpu_count() or 2)
        self.assertEqual(config.queue_capacity, 2000)

class TestDynamicThreadPool(unittest.TestCase):
    """动态线程池功能测试"""

    def setUp(self):
        self.pool = create_default_pool()

    def tearDown(self):
        self.pool.shutdown()

    def test_task_execution(self):
        """测试任务执行"""
        futures = [self.pool.submit(mock_task, i) for i in range(5)]
        results = [f.result() for f in futures]
        self.assertEqual(results, [0, 1, 4, 9, 16])

    def test_pool_stats(self):
        """测试统计信息"""
        stats = self.pool.get_stats()
        self.assertIn('active_threads', stats)
        self.assertIn('pending_tasks', stats)

    @patch('psutil.getloadavg')
    def test_load_monitoring(self, mock_loadavg):
        """测试负载监控"""
        # 模拟高负载
        mock_loadavg.return_value = (os.cpu_count() * 0.8, 0, 0)
        time.sleep(0.1)  # 确保监控线程运行

        # 检查配置是否调整
        stats = self.pool.get_stats()
        self.assertLess(stats['core_pool_size'], os.cpu_count() * 2)

    def test_queue_full(self):
        """测试队列满处理"""
        # 设置小队列进行测试
        config = ThreadPoolConfig(queue_capacity=2)
        small_pool = DynamicThreadPool(config)

        # 提交足够多的任务填满队列
        futures = []
        for i in range(5):
            futures.append(small_pool.submit(mock_task, i))

        # 验证所有任务完成
        results = [f.result() for f in futures]
        self.assertEqual(len(results), 5)

        small_pool.shutdown()

if __name__ == '__main__':
    unittest.main()
