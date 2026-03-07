#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
并行数据加载器测试
测试RQA2025 并行数据加载器的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime, timedelta
from concurrent.futures import Future
class TestParallelLoadingManager(unittest.TestCase):
    """测试并行数据加载管理器"""

    def setUp(self):
        """测试前准备"""
        try:
            import importlib
            import sys
            sys.path.insert(0, 'src')
            module = importlib.import_module('async.data.parallel_loader')
            self.ParallelLoadingManager = module.ParallelLoadingManager
        except ImportError:
            self.skipTest("ParallelLoadingManager not available")

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_initialization(self, mock_thread_pool):
        """测试初始化"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager(max_workers=8)

        self.assertEqual(manager.max_workers, 8)
        self.assertEqual(manager.executor, mock_executor)
        self.assertIsInstance(manager.active_tasks, dict)
        self.assertIsInstance(manager.results_cache, dict)
        mock_thread_pool.assert_called_once_with(max_workers=8)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_initialization_default_workers(self, mock_thread_pool):
        """测试默认工作线程数初始化"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        self.assertEqual(manager.max_workers, 4)
        mock_thread_pool.assert_called_once_with(max_workers=4)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_submit_task_success(self, mock_thread_pool):
        """测试成功提交任务"""
        mock_executor = Mock()
        mock_future = Mock()
        mock_executor.submit.return_value = mock_future
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_loader = Mock()

        with patch('src.async.data.parallel_loader.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)

        manager.submit_task(
            loader=mock_loader,
            task_id="test_task",
            start_date="2023-01-01",
            end_date="2023-01-31",
            frequency="daily"
        )

        self.assertIn("test_task", manager.active_tasks)
        task_info = manager.active_tasks["test_task"]
        self.assertEqual(task_info['future'], mock_future)
        self.assertEqual(task_info['start_time'], datetime(2023, 1, 1, 12, 0, 0))
        self.assertEqual(task_info['params']['start_date'], "2023-01-01")
        self.assertEqual(task_info['params']['end_date'], "2023-01-31")
        self.assertEqual(task_info['params']['frequency'], "daily")

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_submit_task_duplicate(self, mock_thread_pool):
        """测试提交重复任务"""
        mock_executor = Mock()
        mock_future = Mock()
        mock_executor.submit.return_value = mock_future
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_loader = Mock()

        # 第一次提交
        manager.submit_task(
        loader=mock_loader,
        task_id="test_task",
        start_date="2023-01-01",
        end_date="2023-01-31",
        frequency="daily"
        )

        # 第二次提交相同任务ID
        with patch('src.async.data.parallel_loader.logger') as mock_logger:
            manager.submit_task(
                loader=mock_loader,
                task_id="test_task",
                start_date="2023-02-01",
                end_date="2023-02-28",
                frequency="daily"
            )

        mock_logger.warning.assert_called_once_with("Task test_task is already running")
        # 应该只有一次submit调用
        self.assertEqual(mock_executor.submit.call_count, 1)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_get_result_success(self, mock_thread_pool):
        """测试成功获取结果"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_future = Mock()
        mock_future.result.return_value = "test_result"
        mock_result = Mock()

        manager.active_tasks["test_task"] = {
        'future': mock_future,
        'start_time': datetime(2023, 1, 1, 12, 0, 0),
        'params': {'start_date': '2023-01-01'}
        }

        with patch('src.async.data.parallel_loader.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 5, 0)

        result = manager.get_result("test_task")

        self.assertEqual(result, "test_result")
        mock_future.result.assert_called_once_with(timeout=None)

        # 检查任务是否从活跃任务中移除
        self.assertNotIn("test_task", manager.active_tasks)

        # 检查结果是否缓存
        self.assertIn("test_task", manager.results_cache)
        cache_entry = manager.results_cache["test_task"]
        self.assertEqual(cache_entry['result'], "test_result")
        self.assertEqual(cache_entry['completion_time'], datetime(2023, 1, 1, 12, 5, 0))

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_get_result_task_not_found(self, mock_thread_pool):
        """测试获取不存在任务的结果"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        with patch('src.async.data.parallel_loader.logger') as mock_logger:
            result = manager.get_result("nonexistent_task")

        self.assertIsNone(result)
        mock_logger.warning.assert_called_once_with("Task nonexistent_task not found")

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_get_result_with_timeout(self, mock_thread_pool):
        """测试带超时获取结果"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_future = Mock()
        mock_future.result.return_value = "timeout_result"

        manager.active_tasks["test_task"] = {
        'future': mock_future,
        'start_time': datetime(2023, 1, 1, 12, 0, 0),
        'params': {'start_date': '2023-01-01'}
        }

        result = manager.get_result("test_task", timeout=30.0)

        self.assertEqual(result, "timeout_result")
        mock_future.result.assert_called_once_with(timeout=30.0)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_get_result_task_failed(self, mock_thread_pool):
        """测试任务执行失败的结果获取"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_future = Mock()
        mock_future.result.side_effect = Exception("Task execution failed")

        manager.active_tasks["test_task"] = {
        'future': mock_future,
        'start_time': datetime(2023, 1, 1, 12, 0, 0),
        'params': {'start_date': '2023-01-01'}
        }

        with patch('src.async.data.parallel_loader.logger') as mock_logger:
            result = manager.get_result("test_task")

            self.assertIsNone(result)
            mock_logger.error.assert_called_once()
            self.assertIn("Task test_task failed", mock_logger.error.call_args[0][0])

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_wait_all_success(self, mock_thread_pool):
        """测试等待所有任务成功完成"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        # 创建两个模拟任务
        mock_future1 = Mock()
        mock_future1.result.return_value = "result1"
        mock_future2 = Mock()
        mock_future2.result.return_value = "result2"

        manager.active_tasks = {
        "task1": {
            'future': mock_future1,
            'start_time': datetime(2023, 1, 1, 12, 0, 0),
            'params': {'start_date': '2023-01-01'}
        },
        "task2": {
            'future': mock_future2,
            'start_time': datetime(2023, 1, 1, 12, 0, 0),
            'params': {'start_date': '2023-01-01'}
        }
        }

        with patch('src.async.data.parallel_loader.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 5, 0)

            results = manager.wait_all()

            self.assertEqual(results, {"task1": "result1", "task2": "result2"})
            self.assertEqual(len(manager.active_tasks), 0)  # 所有任务都应该被清理
            self.assertEqual(len(manager.results_cache), 2)  # 结果应该被缓存

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_wait_all_with_timeout(self, mock_thread_pool):
        """测试带超时等待所有任务"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_future = Mock()
        mock_future.result.return_value = "timeout_result"

        manager.active_tasks = {
        "task1": {
            'future': mock_future,
            'start_time': datetime(2023, 1, 1, 12, 0, 0),
            'params': {'start_date': '2023-01-01'}
        }
        }

        results = manager.wait_all(timeout=60.0)

        self.assertEqual(results, {"task1": "timeout_result"})
        mock_future.result.assert_called_once_with(timeout=60.0)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_wait_all_task_failure(self, mock_thread_pool):
        """测试等待所有任务时部分任务失败"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        # 一个成功，一个失败
        mock_future1 = Mock()
        mock_future1.result.return_value = "success_result"
        mock_future2 = Mock()
        mock_future2.result.side_effect = Exception("Task failed")

        manager.active_tasks = {
        "task1": {
            'future': mock_future1,
            'start_time': datetime(2023, 1, 1, 12, 0, 0),
            'params': {'start_date': '2023-01-01'}
        },
        "task2": {
            'future': mock_future2,
            'start_time': datetime(2023, 1, 1, 12, 0, 0),
            'params': {'start_date': '2023-01-01'}
        }
        }

        with patch('src.async.data.parallel_loader.logger') as mock_logger:
            with patch('src.async.data.parallel_loader.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 5, 0)

                results = manager.wait_all()

                self.assertEqual(results, {"task1": "success_result"})
                self.assertEqual(len(manager.active_tasks), 0)  # 所有任务都应该被清理
                self.assertEqual(len(manager.results_cache), 1)  # 只有成功的结果被缓存

                # 检查错误日志
                mock_logger.error.assert_called_once()
                self.assertIn("Task task2 failed", mock_logger.error.call_args[0][0])

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_cancel_task_success(self, mock_thread_pool):
        """测试成功取消任务"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_future = Mock()
        mock_future.cancel.return_value = True

        manager.active_tasks["test_task"] = {
        'future': mock_future,
        'start_time': datetime(2023, 1, 1, 12, 0, 0),
        'params': {'start_date': '2023-01-01'}
        }

        with patch('src.async.data.parallel_loader.logger') as mock_logger:
            result = manager.cancel_task("test_task")

            self.assertTrue(result)
            mock_future.cancel.assert_called_once()
            self.assertNotIn("test_task", manager.active_tasks)
            mock_logger.info.assert_called_once_with("Task test_task cancelled")

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_cancel_task_not_found(self, mock_thread_pool):
        """测试取消不存在的任务"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        with patch('src.async.data.parallel_loader.logger') as mock_logger:
            result = manager.cancel_task("nonexistent_task")

            self.assertFalse(result)
            mock_logger.warning.assert_called_once_with("Task nonexistent_task not found")

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_cancel_task_cancel_failed(self, mock_thread_pool):
        """测试取消任务失败"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_future = Mock()
        mock_future.cancel.return_value = False

        manager.active_tasks["test_task"] = {
        'future': mock_future,
        'start_time': datetime(2023, 1, 1, 12, 0, 0),
        'params': {'start_date': '2023-01-01'}
        }

        result = manager.cancel_task("test_task")

        self.assertFalse(result)
        # 任务应该仍然在活跃任务中
        self.assertIn("test_task", manager.active_tasks)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_get_task_status_running(self, mock_thread_pool):
        """测试获取运行中任务的状态"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_future = Mock()
        mock_future.done.return_value = False

        start_time = datetime(2023, 1, 1, 12, 0, 0)
        manager.active_tasks["test_task"] = {
        'future': mock_future,
        'start_time': start_time,
        'params': {'start_date': '2023-01-01', 'frequency': 'daily'}
        }

        status = manager.get_task_status("test_task")

        self.assertEqual(status['status'], 'running')
        self.assertEqual(status['start_time'], start_time.isoformat())
        self.assertEqual(status['params']['start_date'], '2023-01-01')

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_get_task_status_completed(self, mock_thread_pool):
        """测试获取已完成任务的状态"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_future = Mock()
        mock_future.done.return_value = True

        start_time = datetime(2023, 1, 1, 12, 0, 0)
        manager.active_tasks["test_task"] = {
        'future': mock_future,
        'start_time': start_time,
        'params': {'start_date': '2023-01-01', 'frequency': 'daily'}
        }

        status = manager.get_task_status("test_task")

        self.assertEqual(status['status'], 'completed')
        self.assertEqual(status['start_time'], start_time.isoformat())

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_get_task_status_cached(self, mock_thread_pool):
        """测试获取缓存任务的状态"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        completion_time = datetime(2023, 1, 1, 12, 5, 0)

        manager.results_cache["test_task"] = {
        'result': "cached_result",
        'completion_time': completion_time,
        'params': {'start_date': '2023-01-01', 'frequency': 'daily'}
        }

        status = manager.get_task_status("test_task")

        self.assertEqual(status['status'], 'cached')
        self.assertEqual(status['completion_time'], completion_time.isoformat())

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_get_task_status_not_found(self, mock_thread_pool):
        """测试获取不存在任务的状态"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        status = manager.get_task_status("nonexistent_task")

        self.assertEqual(status, {'status': 'not_found'})

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_clear_cache_all(self, mock_thread_pool):
        """测试清理所有缓存"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        manager.results_cache = {
        "task1": {"result": "result1", "completion_time": datetime(2023, 1, 1, 12, 0, 0), "params": {}},
        "task2": {"result": "result2", "completion_time": datetime(2023, 1, 1, 12, 5, 0), "params": {}}
        }

        count = manager.clear_cache()

        self.assertEqual(count, 2)
        self.assertEqual(len(manager.results_cache), 0)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_clear_cache_older_than(self, mock_thread_pool):
        """测试清理指定时间之前的缓存"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        old_time = datetime(2023, 1, 1, 11, 0, 0)
        new_time = datetime(2023, 1, 1, 13, 0, 0)
        cutoff_time = datetime(2023, 1, 1, 12, 0, 0)

        manager.results_cache = {
        "old_task": {"result": "old", "completion_time": old_time, "params": {}},
        "new_task": {"result": "new", "completion_time": new_time, "params": {}}
        }

        count = manager.clear_cache(older_than=cutoff_time)

        self.assertEqual(count, 1)
        self.assertNotIn("old_task", manager.results_cache)
        self.assertIn("new_task", manager.results_cache)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_shutdown_with_wait(self, mock_thread_pool):
        """测试带等待的关闭"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        with patch('src.async.data.parallel_loader.logger') as mock_logger:
            manager.shutdown(wait=True)

            mock_executor.shutdown.assert_called_once_with(wait=True)
            mock_logger.info.assert_called_once_with("ParallelLoadingManager shutdown complete")

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_shutdown_without_wait(self, mock_thread_pool):
        """测试不带等待的关闭"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        manager.shutdown(wait=False)

        mock_executor.shutdown.assert_called_once_with(wait=False)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_shutdown_error(self, mock_thread_pool):
        """测试关闭时的错误"""
        mock_executor = Mock()
        mock_executor.shutdown.side_effect = Exception("Shutdown error")
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        with patch('src.async.data.parallel_loader.logger') as mock_logger:
            manager.shutdown()

        mock_logger.error.assert_called_once()
        self.assertIn("关闭并行加载管理器失败", mock_logger.error.call_args[0][0])
class TestParallelLoadingManagerIntegration(unittest.TestCase):
    """测试并行数据加载管理器集成"""

    def setUp(self):
        """测试前准备"""
        try:
            import importlib
            import sys
            sys.path.insert(0, 'src')
            module = importlib.import_module('async.data.parallel_loader')
            self.ParallelLoadingManager = module.ParallelLoadingManager
        except ImportError:
            self.skipTest("ParallelLoadingManager not available")

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_load_data_parallel_success(self, mock_thread_pool):
        """测试并行数据加载成功"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        # Mock futures for parallel execution
        mock_futures = [Mock() for _ in range(2)]
        mock_futures[0].result.return_value = "data1"
        mock_futures[1].result.return_value = "data2"

        mock_executor.submit.side_effect = mock_futures

        with patch('src.async.data.parallel_loader.as_completed') as mock_as_completed:
            mock_as_completed.return_value = mock_futures

        symbols = ["AAPL", "GOOGL"]
        results = manager.load_data_parallel(
            data_type="stock",
            start_date="2023-01-01",
            end_date="2023-01-31",
            frequency="daily",
            symbols=symbols
        )

        self.assertEqual(results, {"AAPL": "data1", "GOOGL": "data2"})
        self.assertEqual(mock_executor.submit.call_count, 2)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_load_data_parallel_partial_failure(self, mock_thread_pool):
        """测试并行数据加载部分失败"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        # Mock futures with one success and one failure
        mock_future_success = Mock()
        mock_future_success.result.return_value = "success_data"
        mock_future_failure = Mock()
        mock_future_failure.result.side_effect = Exception("Load failed")

        mock_executor.submit.side_effect = [mock_future_success, mock_future_failure]

        with patch('src.async.data.parallel_loader.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future_success, mock_future_failure]

            symbols = ["AAPL", "GOOGL"]
            with patch('src.async.data.parallel_loader.logger') as mock_logger:
                results = manager.load_data_parallel(
                data_type="stock",
                start_date="2023-01-01",
                end_date="2023-01-31",
                frequency="daily",
                symbols=symbols
            )

            self.assertEqual(results, {"AAPL": "success_data", "GOOGL": None})
            mock_logger.error.assert_called_once()

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_load_data_parallel_empty_symbols(self, mock_thread_pool):
        """测试并行数据加载空股票列表"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        results = manager.load_data_parallel(
            data_type="stock",
            start_date="2023-01-01",
            end_date="2023-01-31",
        frequency="daily",
        symbols=[]
        )

        self.assertEqual(results, {})
        mock_executor.submit.assert_not_called()
class TestParallelLoadingManagerEdgeCases(unittest.TestCase):
    """测试并行数据加载管理器边界情况"""

    def setUp(self):
        """测试前准备"""
        try:
            import importlib
            import sys
            sys.path.insert(0, 'src')
            module = importlib.import_module('async.data.parallel_loader')
            self.ParallelLoadingManager = module.ParallelLoadingManager
        except ImportError:
            self.skipTest("ParallelLoadingManager not available")

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_zero_max_workers(self, mock_thread_pool):
        """测试零工作线程"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager(max_workers=0)

        self.assertEqual(manager.max_workers, 0)
        mock_thread_pool.assert_called_once_with(max_workers=0)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_large_max_workers(self, mock_thread_pool):
        """测试大量工作线程"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager(max_workers=100)

        self.assertEqual(manager.max_workers, 100)
        mock_thread_pool.assert_called_once_with(max_workers=100)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_get_result_after_cache_clear(self, mock_thread_pool):
        """测试清理缓存后获取结果"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()

        # 先缓存一个结果
        manager.results_cache["test_task"] = {
        'result': "cached_result",
        'completion_time': datetime(2023, 1, 1, 12, 0, 0),
        'params': {'start_date': '2023-01-01'}
        }

        # 清理缓存
        manager.clear_cache()

        # 获取状态应该返回not_found
        status = manager.get_task_status("test_task")
        self.assertEqual(status, {'status': 'not_found'})

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_multiple_operations_on_same_task(self, mock_thread_pool):
        """测试对同一任务的多次操作"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_future = Mock()
        mock_future.result.return_value = "test_result"

        manager.active_tasks["test_task"] = {
        'future': mock_future,
        'start_time': datetime(2023, 1, 1, 12, 0, 0),
        'params': {'start_date': '2023-01-01'}
        }

        # 第一次获取结果
        with patch('src.async.data.parallel_loader.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 5, 0)

            result1 = manager.get_result("test_task")
            self.assertEqual(result1, "test_result")

            # 任务应该已被移除
            self.assertNotIn("test_task", manager.active_tasks)

            # 再次获取应该返回None
            result2 = manager.get_result("test_task")
            self.assertIsNone(result2)

    @patch('src.async.data.parallel_loader.ThreadPoolExecutor')
    def test_task_status_transitions(self, mock_thread_pool):
        """测试任务状态转换"""
        mock_executor = Mock()
        mock_thread_pool.return_value = mock_executor

        manager = self.ParallelLoadingManager()
        mock_future = Mock()

        # 初始状态：运行中
        mock_future.done.return_value = False
        manager.active_tasks["test_task"] = {
        'future': mock_future,
        'start_time': datetime(2023, 1, 1, 12, 0, 0),
        'params': {'start_date': '2023-01-01'}
        }

        status = manager.get_task_status("test_task")
        self.assertEqual(status['status'], 'running')

        # 状态转换：完成
        mock_future.done.return_value = True
        status = manager.get_task_status("test_task")
        self.assertEqual(status['status'], 'completed')

        # 手动模拟结果缓存
        manager.results_cache["test_task"] = {
        'result': "completed_result",
        'completion_time': datetime(2023, 1, 1, 12, 5, 0),
        'params': {'start_date': '2023-01-01'}
        }
        # 删除活跃任务
        del manager.active_tasks["test_task"]

        # 状态转换：缓存
        status = manager.get_task_status("test_task")
        self.assertEqual(status['status'], 'cached')


if __name__ == '__main__':
    unittest.main()
