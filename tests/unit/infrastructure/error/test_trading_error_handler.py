import unittest
import threading
from src.infrastructure.error.trading_error_handler import TradingErrorHandler
from src.infrastructure.error.exceptions import TradingError

class TestTradingErrorHandler(unittest.TestCase):
    """测试交易错误处理器"""

    def setUp(self):
        self.handler = TradingErrorHandler(
            log_errors=False,  # 测试时不记录日志
            raise_unknown=True,
            max_history_size=100
        )

    def test_thread_safe_error_handling(self):
        """测试线程安全的错误处理"""
        results = []
        threads = []

        def worker(error_type):
            try:
                result = self.handler.handle_order_error(
                    error_type,
                    {'message': 'Test error', 'context': {'thread': threading.get_ident()}}
                )
                results.append(result)
            except Exception as e:
                results.append(str(e))

        # 创建多个线程并发测试不同错误类型
        error_types = [
            TradingError.ORDER_REJECTED,
            TradingError.CONNECTION_ERROR,
            TradingError.MARKET_CLOSED
        ]
        for error_type in error_types:
            t = threading.Thread(target=worker, args=(error_type,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 验证所有线程都处理了错误
        self.assertEqual(len(results), 3)

    def test_error_history_limits(self):
        """测试错误历史记录限制"""
        for i in range(150):  # 超过最大历史记录数
            self.handler.handle_order_error(
                TradingError.ORDER_REJECTED,
                {'message': f'Error {i}', 'context': {'count': i}}
            )

        stats = self.handler.get_trading_error_stats()
        self.assertEqual(stats['total_errors'], 100)  # 验证限制生效
        self.assertEqual(stats['by_type']['ORDER_REJECTED'], 100)

    def test_error_stats_thread_breakdown(self):
        """测试按线程统计错误"""
        main_thread_id = threading.get_ident()

        # 模拟主线程错误
        self.handler.handle_order_error(
            TradingError.ORDER_REJECTED,
            {'message': 'Main thread error'}
        )

        # 模拟工作线程错误
        def worker():
            self.handler.handle_order_error(
                TradingError.CONNECTION_ERROR,
                {'message': 'Worker thread error'}
            )

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        stats = self.handler.get_trading_error_stats()
        self.assertEqual(stats['thread_stats'][str(main_thread_id)], 1)
        self.assertEqual(stats['thread_stats'][str(t.ident)], 1)

    def test_exception_context_preservation(self):
        """测试异常上下文保留"""
        test_context = {'order_id': '12345', 'symbol': 'AAPL'}

        with self.assertRaises(TradingError) as cm:
            self.handler.handle_order_error(
                TradingError.INSUFFICIENT_FUNDS,
                {'message': 'Test', 'context': test_context}
            )

        # 验证异常中保留了上下文
        self.assertIn('12345', str(cm.exception))
        self.assertIn('AAPL', str(cm.exception))

if __name__ == '__main__':
    unittest.main()
