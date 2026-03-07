"""
基础设施层 - 策略组件单元测试

测试CircuitBreaker和RetryPolicy的核心功能。
覆盖率目标: 85%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.error.policies.circuit_breaker import CircuitBreaker, CircuitBreakerState
from src.infrastructure.error.policies.retry_policy import RetryPolicy, RetryStrategy


class TestCircuitBreaker(unittest.TestCase):
    """CircuitBreaker 单元测试"""

    def setUp(self):
        """测试前准备"""
        self.breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5.0,
            success_threshold=2
        )

    def test_initialization(self):
        """测试初始化"""
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=10.0)

        self.assertEqual(breaker.failure_threshold, 5)
        self.assertEqual(breaker.recovery_timeout, 10.0)
        self.assertEqual(breaker.success_threshold, 2)
        self.assertEqual(breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(breaker.failure_count, 0)
        self.assertEqual(breaker.success_count, 0)
        self.assertIsNone(breaker.last_failure_time)

    def test_initial_closed_state(self):
        """测试初始闭合状态"""
        self.assertEqual(self.breaker.state, CircuitBreakerState.CLOSED)
        self.assertTrue(self.breaker.is_closed())
        self.assertFalse(self.breaker.is_open())
        self.assertFalse(self.breaker.is_half_open())

    def test_successful_calls(self):
        """测试成功调用"""
        # 执行多次成功调用
        for i in range(5):
            self.breaker.record_success()

        # 验证状态保持闭合
        self.assertTrue(self.breaker.is_closed())
        self.assertEqual(self.breaker.failure_count, 0)
        self.assertEqual(self.breaker.success_count, 0)  # 成功计数在闭合状态下不累积

    def test_failure_calls_open_circuit(self):
        """测试失败调用导致熔断器打开"""
        # 执行失败调用达到阈值
        for i in range(3):
            self.breaker.record_failure()

        # 验证熔断器打开
        self.assertTrue(self.breaker.is_open())
        self.assertEqual(self.breaker.failure_count, 3)
        self.assertIsNotNone(self.breaker.last_failure_time)

    def test_failure_calls_below_threshold(self):
        """测试失败调用未达到阈值"""
        # 执行少量失败调用
        for i in range(2):
            self.breaker.record_failure()

        # 验证熔断器仍闭合
        self.assertTrue(self.breaker.is_closed())
        self.assertEqual(self.breaker.failure_count, 2)

    def test_half_open_after_timeout(self):
        """测试超时后进入半开状态"""
        # 触发熔断器打开
        for i in range(3):
            self.breaker.record_failure()

        self.assertTrue(self.breaker.is_open())

        # 模拟时间流逝超过恢复超时
        self.breaker.last_failure_time = time.time() - 6.0  # 超过5秒超时

        # 手动检查状态转换（模拟内部逻辑）
        if self.breaker.is_open() and self.breaker._should_attempt_reset():
            self.breaker.half_open()

        self.assertTrue(self.breaker.is_half_open())

    def test_recovery_from_half_open_success(self):
        """测试从半开状态成功恢复"""
        # 进入半开状态
        self.breaker.half_open()
        self.breaker.success_count = 0

        # 记录足够成功的调用
        for i in range(2):
            self.breaker.record_success()

        # 验证熔断器关闭
        self.assertTrue(self.breaker.is_closed())
        self.assertEqual(self.breaker.success_count, 0)  # 重置成功计数
        self.assertEqual(self.breaker.failure_count, 0)  # 重置失败计数

    def test_recovery_from_half_open_failure(self):
        """测试从半开状态恢复失败"""
        # 进入半开状态
        self.breaker.half_open()

        # 记录失败调用
        self.breaker.record_failure()

        # 验证熔断器重新打开
        self.assertTrue(self.breaker.is_open())
        self.assertEqual(self.breaker.success_count, 0)  # 重置成功计数

    def test_manual_state_transitions(self):
        """测试手动状态转换"""
        # 手动打开熔断器
        self.breaker.open()
        self.assertTrue(self.breaker.is_open())

        # 手动关闭熔断器
        self.breaker.close()
        self.assertTrue(self.breaker.is_closed())

        # 手动设置为半开
        self.breaker.half_open()
        self.assertTrue(self.breaker.is_half_open())

    def test_reset_functionality(self):
        """测试重置功能"""
        # 累积一些失败
        for i in range(2):
            self.breaker.record_failure()

        # 重置
        self.breaker.reset()

        # 验证状态重置
        self.assertTrue(self.breaker.is_closed())
        self.assertEqual(self.breaker.failure_count, 0)
        self.assertEqual(self.breaker.success_count, 0)
        self.assertIsNone(self.breaker.last_failure_time)

    def test_call_permitted_in_closed_state(self):
        """测试闭合状态下允许调用"""
        self.assertTrue(self.breaker.call_permitted())

    def test_call_permitted_in_open_state(self):
        """测试打开状态下禁止调用"""
        self.breaker.open()
        self.assertFalse(self.breaker.call_permitted())

    def test_call_permitted_in_half_open_state(self):
        """测试半开状态下允许调用"""
        self.breaker.half_open()
        self.assertTrue(self.breaker.call_permitted())

    def test_get_status(self):
        """测试获取统计信息"""
        # 执行一些操作
        self.breaker.record_success()
        self.breaker.record_failure()
        self.breaker.record_failure()

        stats = self.breaker.get_status()

        # 验证统计信息结构
        self.assertIn('state', stats)
        self.assertIn('failure_count', stats)
        self.assertIn('success_count', stats)
        self.assertIn('failure_threshold', stats)
        self.assertIn('recovery_timeout', stats)
        self.assertIn('success_threshold', stats)
        self.assertIn('last_failure_time', stats)

        # 验证数据正确性
        self.assertEqual(stats['state'], 'closed')
        self.assertEqual(stats['failure_count'], 2)
        self.assertEqual(stats['success_count'], 0)
        self.assertEqual(stats['failure_threshold'], 3)

    def test_state_persistence_simulation(self):
        """测试状态持久化模拟"""
        # 这里我们不实现真正的持久化，但测试状态转换的逻辑
        initial_state = self.breaker.state

        # 执行状态转换
        self.breaker.open()
        open_state = self.breaker.state

        self.breaker.close()
        closed_state = self.breaker.state

        # 验证状态转换正确
        self.assertNotEqual(initial_state, open_state)
        self.assertEqual(initial_state, closed_state)

    def test_concurrent_access_simulation(self):
        """测试并发访问模拟"""
        import threading

        results = []
        errors = []

        def breaker_worker(worker_id):
            """熔断器工作线程"""
            try:
                for i in range(10):
                    if self.breaker.call_permitted():
                        if (worker_id + i) % 3 == 0:
                            self.breaker.record_failure()
                        else:
                            self.breaker.record_success()
                    results.append((worker_id, i, self.breaker.state.name))
                    time.sleep(0.001)
            except Exception as e:
                errors.append((worker_id, str(e)))

        # 启动并发线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=breaker_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(errors), 0, f"并发操作出现错误: {errors}")
        self.assertEqual(len(results), 30)  # 3线程 * 10次操作


class TestRetryPolicy(unittest.TestCase):
    """RetryPolicy 单元测试"""

    def setUp(self):
        """测试前准备"""
        self.policy = RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            strategy=RetryStrategy.FIXED
        )

    def test_initialization(self):
        """测试初始化"""
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=2.0,
            strategy=RetryStrategy.EXPONENTIAL,
            backoff_factor=2.0,
            max_delay=60.0,
            jitter=True
        )

        self.assertEqual(policy.max_attempts, 5)
        self.assertEqual(policy.base_delay, 2.0)
        self.assertEqual(policy.strategy, RetryStrategy.EXPONENTIAL)
        self.assertEqual(policy.backoff_factor, 2.0)
        self.assertEqual(policy.max_delay, 60.0)
        self.assertTrue(policy.jitter)

    def test_fixed_delay_strategy(self):
        """测试固定延迟策略"""
        policy = RetryPolicy(strategy=RetryStrategy.FIXED, base_delay=1.0, jitter=False)

        # 验证延迟计算
        for attempt in range(5):
            delay = policy.calculate_delay(attempt)
            self.assertEqual(delay, 1.0)

    def test_linear_delay_strategy(self):
        """测试线性延迟策略"""
        policy = RetryPolicy(strategy=RetryStrategy.LINEAR, base_delay=1.0, jitter=False)

        expected_delays = [1.0, 2.0, 3.0, 4.0, 5.0]  # (attempt + 1) * base_delay

        for attempt, expected in enumerate(expected_delays):
            delay = policy.calculate_delay(attempt)
            self.assertEqual(delay, expected)

    def test_exponential_delay_strategy(self):
        """测试指数延迟策略"""
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            backoff_factor=2.0,
            jitter=False
        )

        expected_delays = [1.0, 2.0, 4.0, 8.0, 16.0]  # base_delay * (backoff_factor ** attempt)

        for attempt, expected in enumerate(expected_delays):
            delay = policy.calculate_delay(attempt)
            self.assertEqual(delay, expected)

    def test_max_delay_limit(self):
        """测试最大延迟限制"""
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=10.0,
            backoff_factor=10.0,
            max_delay=50.0,
            jitter=False
        )

        # 高尝试次数应该被限制
        delay = policy.calculate_delay(5)  # 10.0 * (10.0 ** 5) = 100000.0
        self.assertEqual(delay, 50.0)  # 应该被限制为最大值

    def test_jitter_effect(self):
        """测试抖动效果"""
        policy = RetryPolicy(strategy=RetryStrategy.FIXED, base_delay=1.0, jitter=True)

        delays = []
        for i in range(10):
            delay = policy.calculate_delay(0)
            delays.append(delay)

        # 验证抖动范围（0.5-1.5倍的基础延迟）
        for delay in delays:
            self.assertGreaterEqual(delay, 0.5)
            self.assertLessEqual(delay, 1.5)

    def test_should_retry_success(self):
        """测试应该重试 - 成功情况"""
        attempt = 1
        exception = ValueError("临时错误")

        should_retry = self.policy.should_retry(attempt, exception)
        self.assertTrue(should_retry)

    def test_should_retry_max_attempts_exceeded(self):
        """测试应该重试 - 超过最大尝试次数"""
        attempt = 3  # 等于max_attempts
        exception = ValueError("错误")

        should_retry = self.policy.should_retry(attempt, exception)
        self.assertFalse(should_retry)

    def test_should_retry_non_retryable_exception(self):
        """测试应该重试 - 不可重试异常"""
        attempt = 1

        # 测试一些常见的不可重试异常
        non_retryable_exceptions = [
            KeyboardInterrupt("用户中断"),
            SystemExit("系统退出"),
            MemoryError("内存不足"),
        ]

        for exception in non_retryable_exceptions:
            should_retry = self.policy.should_retry(attempt, exception)
            self.assertFalse(should_retry, f"异常 {type(exception).__name__} 应该不可重试")

    def test_retry_execution_success(self):
        """测试重试执行成功"""
        call_count = 0

        def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("第一次失败")
            return "success"

        result = self.policy.execute(mock_operation)

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)  # 第一次失败，第二次成功

    def test_retry_execution_failure(self):
        """测试重试执行失败"""
        def always_failing_operation():
            raise ValueError("总是失败")

        with self.assertRaises(ValueError):
            self.policy.execute(always_failing_operation)

    def test_retry_with_custom_should_retry(self):
        """测试带自定义重试判断的重试"""
        policy = RetryPolicy(max_attempts=3)

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("可重试错误")
            return "success"

        result = policy.execute(operation)

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)  # 第一次失败，第二次失败，第三次成功

    def test_retry_with_context(self):
        """测试带上下文的重试"""
        context = {"user_id": 123, "operation": "test"}

        call_count = 0

        def context_aware_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError(f"用户 {context['user_id']} 操作失败")
            return f"用户 {context['user_id']} 操作成功"

        result = self.policy.execute(context_aware_operation)

        self.assertEqual(result, "用户 123 操作成功")
        self.assertEqual(call_count, 2)

    def test_get_retry_stats(self):
        """测试获取重试统计信息"""
        stats = self.policy.get_retry_stats()

        # 验证统计信息结构
        self.assertIn('max_attempts', stats)
        self.assertIn('base_delay', stats)
        self.assertIn('max_delay', stats)
        self.assertIn('strategy', stats)
        self.assertIn('jitter_enabled', stats)
        self.assertIn('backoff_factor', stats)

        # 验证数据正确性
        self.assertEqual(stats['max_attempts'], 3)
        self.assertEqual(stats['base_delay'], 1.0)
        self.assertEqual(stats['strategy'], 'fixed')

    def test_reset_stats(self):
        """测试重置统计信息"""
        # 重置统计
        self.policy.reset_stats()
        stats_after = self.policy.get_retry_stats()

        # 验证统计被重置（对于这个实现，重置统计实际上不做任何事情）
        self.assertIn('max_attempts', stats_after)

    def test_different_strategies_comparison(self):
        """测试不同策略的比较"""
        strategies = [
            RetryStrategy.FIXED,
            RetryStrategy.LINEAR,
            RetryStrategy.EXPONENTIAL
        ]

        for strategy in strategies:
            policy = RetryPolicy(strategy=strategy, base_delay=1.0, max_attempts=1)

            # 验证策略设置正确
            self.assertEqual(policy.strategy, strategy)

            # 验证延迟计算不抛出异常
            try:
                delay = policy.calculate_delay(0)
                self.assertIsInstance(delay, float)
                self.assertGreaterEqual(delay, 0)
            except Exception as e:
                self.fail(f"策略 {strategy.value} 计算延迟失败: {e}")

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试attempt为0的情况
        delay = self.policy.calculate_delay(0)
        self.assertGreaterEqual(delay, 0)

        # 测试最大attempt的情况
        max_attempt_policy = RetryPolicy(max_attempts=1)
        should_retry = max_attempt_policy.should_retry(1, ValueError("error"))
        self.assertFalse(should_retry)

        # 测试延迟为0的情况
        zero_delay_policy = RetryPolicy(base_delay=0.0)
        delay = zero_delay_policy.calculate_delay(0)
        self.assertEqual(delay, 0.0)

    def test_thread_safety_simulation(self):
        """测试线程安全性模拟"""
        import threading

        results = []
        errors = []

        def retry_worker(worker_id):
            """重试工作线程"""
            try:
                call_count = 0

                def worker_operation():
                    nonlocal call_count
                    call_count += 1
                    if call_count < 2:
                        raise ValueError(f"Worker {worker_id} failure")
                    return f"Worker {worker_id} success"

                result = self.policy.execute(worker_operation)
                results.append((worker_id, result, call_count))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # 启动并发线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=retry_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(errors), 0, f"并发重试出现错误: {errors}")
        self.assertEqual(len(results), 3)

        # 验证每个worker都成功了
        for worker_id, result, call_count in results:
            self.assertIn(f"Worker {worker_id} success", result)
            self.assertEqual(call_count, 2)  # 每次都重试1次


if __name__ == '__main__':
    unittest.main()