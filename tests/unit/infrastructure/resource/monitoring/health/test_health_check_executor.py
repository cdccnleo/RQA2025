"""
测试目标：提升resource/monitoring/health/health_check_executor.py的真实覆盖率
实际导入和使用src.infrastructure.resource.monitoring.health.health_check_executor模块
"""

import time
import threading
from unittest.mock import Mock, patch
import pytest

from src.infrastructure.resource.monitoring.health.health_check_executor import HealthCheckExecutor
from src.infrastructure.resource.monitoring.health.health_check_manager import HealthCheck


class TestHealthCheckExecutor:
    """测试HealthCheckExecutor类"""

    @pytest.fixture
    def mock_logger(self):
        """模拟logger"""
        return Mock()

    @pytest.fixture
    def executor(self, mock_logger):
        """创建执行器实例"""
        return HealthCheckExecutor(mock_logger)

    def test_initialization(self, executor, mock_logger):
        """测试初始化"""
        assert executor.logger == mock_logger
        assert isinstance(executor._lock, type(threading.RLock()))

    def test_initialization_without_logger(self):
        """测试不提供logger时的初始化"""
        executor = HealthCheckExecutor()
        assert executor.logger is not None
        assert hasattr(executor.logger, 'log_error')

    def test_execute_check_success(self, executor, mock_logger):
        """测试成功执行检查"""
        check = Mock(spec=HealthCheck)
        check.name = "test_check"
        check.check_function = Mock(return_value={"status": "healthy"})
        check.timeout = 30

        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0

            result = executor.execute_check(check)

            assert result['check_name'] == "test_check"
            assert result['status'] == 'success'
            assert result['result'] == {"status": "healthy"}
            assert result['execution_time'] > 0
            assert result['timestamp'] == 1000.0
            assert result['error'] is None

    def test_execute_check_failure(self, executor, mock_logger):
        """测试执行检查失败"""
        check = Mock(spec=HealthCheck)
        check.name = "test_check"
        check.check_function = Mock(side_effect=Exception("Check failed"))
        check.timeout = 30

        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0

            result = executor.execute_check(check)

            assert result['check_name'] == "test_check"
            assert result['status'] == 'failed'
            assert result['result'] is None
            assert result['execution_time'] > 0
            assert result['timestamp'] == 1000.0
            assert 'Check failed' in str(result['error'])

    def test_execute_check_timeout(self, executor, mock_logger):
        """测试执行检查超时"""
        check = Mock(spec=HealthCheck)
        check.name = "test_check"
        check.check_function = Mock(side_effect=TimeoutError("Timeout"))
        check.timeout = 30

        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0

            result = executor.execute_check(check)

            assert result['check_name'] == "test_check"
            assert result['status'] == 'failed'
            assert result['result'] is None
            assert 'Timeout' in str(result['error'])

    def test_execute_multiple_checks(self, executor, mock_logger):
        """测试执行多个检查"""
        check1 = Mock(spec=HealthCheck)
        check1.name = "check1"
        check1.check_function = Mock(return_value={"status": "healthy"})
        check1.timeout = 30

        check2 = Mock(spec=HealthCheck)
        check2.name = "check2"
        check2.check_function = Mock(side_effect=Exception("Failed"))
        check2.timeout = 30

        results = executor.execute_multiple_checks([check1, check2])

        assert len(results) == 2
        assert results[0]['check_name'] == "check1"
        assert results[0]['status'] == 'success'
        assert results[1]['check_name'] == "check2"
        assert results[1]['status'] == 'failed'

    def test_execute_multiple_checks_empty_list(self, executor):
        """测试执行空检查列表"""
        results = executor.execute_multiple_checks([])

        assert results == []

    def test_get_executor_status(self, executor):
        """测试获取执行器状态"""
        status = executor.get_executor_status()

        assert 'thread_safe' in status
        assert 'lock_available' in status
        assert status['thread_safe'] is True

    def test__execute_with_timeout_success(self, executor):
        """测试带超时的执行成功"""
        def success_func():
            return "success"

        result = executor._execute_with_timeout(success_func, 5)

        assert result == "success"

    def test__execute_with_timeout_timeout(self, executor):
        """测试带超时的执行超时"""
        def slow_func():
            time.sleep(2)
            return "slow"

        with pytest.raises(TimeoutError):
            executor._execute_with_timeout(slow_func, 0.1)

    def test__execute_with_timeout_exception(self, executor):
        """测试带超时的执行异常"""
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            executor._execute_with_timeout(failing_func, 5)

    def test_get_check_statistics(self, executor):
        """测试获取检查统计"""
        # 执行一些检查来生成统计
        check = Mock(spec=HealthCheck)
        check.name = "test_check"
        check.check_function = Mock(return_value={"status": "healthy"})
        check.timeout = 30

        executor.execute_check(check)
        executor.execute_check(check)

        stats = executor.get_check_statistics()

        assert 'total_checks' in stats
        assert 'successful_checks' in stats
        assert 'failed_checks' in stats
        assert 'average_execution_time' in stats
        assert stats['total_checks'] >= 2

    def test_get_check_statistics_no_checks(self, executor):
        """测试获取无检查时的统计"""
        stats = executor.get_check_statistics()

        assert stats['total_checks'] == 0
        assert stats['successful_checks'] == 0
        assert stats['failed_checks'] == 0
        assert stats['average_execution_time'] == 0.0

    def test_thread_safety(self, executor):
        """测试线程安全性"""
        results = []
        errors = []

        def worker():
            try:
                check = Mock(spec=HealthCheck)
                check.name = f"thread_check_{threading.current_thread().ident}"
                check.check_function = Mock(return_value={"status": "healthy"})
                check.timeout = 30

                result = executor.execute_check(check)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 5
        assert len(errors) == 0
        for result in results:
            assert result['status'] == 'success'
























