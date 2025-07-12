import unittest
import threading
import time
from src.infrastructure.error.retry_handler import RetryHandler
from src.infrastructure.error.exceptions import RetryError

class TestRetryHandler(unittest.TestCase):
    """测试重试处理器"""

    def setUp(self):
        self.retry_handler = RetryHandler(
            initial_delay=0.1,
            max_delay=1.0,
            backoff_factor=2,
            jitter=0.1,
            max_attempts=3
        )

    def test_thread_safety(self):
        """测试线程安全性"""
        results = []
        threads = []
        results_lock = threading.Lock()

        def worker():
            retry_handler = RetryHandler(
                initial_delay=0.1,
                max_delay=1.0,
                backoff_factor=2,
                jitter=0.1,
                max_attempts=3
            )
            @retry_handler.with_retry()
            def always_42():
                return 42
            try:
                result = always_42()
                with results_lock:
                    results.append(result)
                print(f"[Thread {threading.get_ident()}] result: {result}")
            except Exception as e:
                with results_lock:
                    results.append(str(e))
                print(f"[Thread {threading.get_ident()}] exception: {e}")

        # 创建多个线程并发测试
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 验证所有线程都成功执行
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r == 42 for r in results))

    def test_retry_history_recording(self):
        """测试重试历史记录"""
        @self.retry_handler.with_retry()
        def failing_func():
            raise ValueError("Intentional error")

        try:
            failing_func()
        except RetryError:
            pass

        history = self.retry_handler.get_retry_history()
        self.assertEqual(len(history), 3)  # 3次尝试
        self.assertEqual(history[0]['attempt'], 1)
        self.assertEqual(history[-1]['attempt'], 3)

    def test_exception_chaining(self):
        """测试异常链"""
        @self.retry_handler.with_retry()
        def failing_func():
            raise ValueError("Original error")

        with self.assertRaises(RetryError) as cm:
            failing_func()

        self.assertIsInstance(cm.exception.__cause__, ValueError)
        self.assertEqual(str(cm.exception.__cause__), "Original error")

    def test_performance_under_load(self):
        """测试性能基准"""
        start_time = time.time()

        @self.retry_handler.with_retry()
        def fast_func():
            return True

        # 执行1000次快速函数
        for _ in range(1000):
            fast_func()

        elapsed = time.time() - start_time
        self.assertLess(elapsed, 1.0)  # 应在1秒内完成

if __name__ == '__main__':
    unittest.main()
