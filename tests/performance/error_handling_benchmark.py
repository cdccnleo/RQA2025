import time
import threading
from src.infrastructure.error.retry_handler import RetryHandler
from src.infrastructure.error.trading_error_handler import TradingErrorHandler
from src.infrastructure.error.exceptions import TradingError

def benchmark_retry_handler():
    """重试处理器性能基准测试"""
    handler = RetryHandler(
        initial_delay=0.01,
        max_delay=0.1,
        backoff_factor=1.5,
        jitter=0.05,
        max_attempts=3
    )

    @handler.with_retry()
    def successful_operation():
        return True

    @handler.with_retry()
    def failing_operation():
        raise ValueError("Intentional error")

    # 测试成功操作性能
    start = time.time()
    for _ in range(1000):
        successful_operation()
    success_duration = time.time() - start

    # 测试失败操作性能
    start = time.time()
    for _ in range(100):
        try:
            failing_operation()
        except:
            pass
    fail_duration = time.time() - start

    print(f"\nRetryHandler 性能基准:")
    print(f"- 1000次成功操作: {success_duration:.3f}秒")
    print(f"- 100次失败操作(3次重试): {fail_duration:.3f}秒")

def benchmark_trading_error_handler():
    """交易错误处理器性能基准测试"""
    handler = TradingErrorHandler(
        log_errors=False,
        raise_unknown=False,
        max_history_size=1000
    )

    # 测试不同错误类型的处理性能
    error_types = [
        TradingError.ORDER_REJECTED,
        TradingError.CONNECTION_ERROR,
        TradingError.MARKET_CLOSED
    ]

    def worker(error_type):
        for _ in range(100):
            handler.handle_order_error(
                error_type,
                {'message': 'Benchmark error', 'context': {'test': True}}
            )

    print("\nTradingErrorHandler 性能基准:")

    # 单线程性能
    for error_type in error_types:
        start = time.time()
        worker(error_type)
        duration = time.time() - start
        print(f"- 100次 {error_type.name} 处理: {duration:.3f}秒")

    # 多线程性能
    threads = []
    start = time.time()
    for error_type in error_types:
        t = threading.Thread(target=worker, args=(error_type,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    concurrent_duration = time.time() - start

    print(f"- 300次并发错误处理(3线程): {concurrent_duration:.3f}秒")

if __name__ == '__main__':
    benchmark_retry_handler()
    benchmark_trading_error_handler()
