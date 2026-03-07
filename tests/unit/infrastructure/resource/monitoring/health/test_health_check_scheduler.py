"""
测试目标：提升resource/monitoring/health/health_check_scheduler.py的真实覆盖率
实际导入和使用src.infrastructure.resource.monitoring.health.health_check_scheduler模块
"""

import time
import threading
from unittest.mock import Mock, patch
import pytest

from src.infrastructure.resource.monitoring.health.health_check_scheduler import HealthCheckScheduler
from src.infrastructure.resource.monitoring.health.health_check_executor import HealthCheckExecutor
from src.infrastructure.resource.monitoring.health.health_check_manager import HealthCheck


class TestHealthCheckScheduler:
    """测试HealthCheckScheduler类"""

    @pytest.fixture
    def mock_executor(self):
        """模拟HealthCheckExecutor"""
        return Mock(spec=HealthCheckExecutor)

    @pytest.fixture
    def mock_logger(self):
        """模拟logger"""
        return Mock()

    @pytest.fixture
    def scheduler(self, mock_executor, mock_logger):
        """创建调度器实例"""
        return HealthCheckScheduler(mock_executor, mock_logger)

    def test_initialization(self, scheduler, mock_executor, mock_logger):
        """测试初始化"""
        assert scheduler.executor == mock_executor
        assert scheduler.logger == mock_logger
        assert scheduler._running is False
        assert scheduler._scheduler_thread is None
        assert isinstance(scheduler._check_threads, dict)
        assert isinstance(scheduler._last_execution, dict)
        assert isinstance(scheduler._lock, type(threading.RLock()))

    def test_initialization_without_logger(self, mock_executor):
        """测试不提供logger时的初始化"""
        scheduler = HealthCheckScheduler(mock_executor)
        assert scheduler.executor == mock_executor
        assert scheduler.logger is not None
        assert hasattr(scheduler.logger, 'log_info')

    def test_start_scheduler_success(self, scheduler, mock_logger):
        """测试成功启动调度器"""
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            result = scheduler.start_scheduler()

            assert result is True
            assert scheduler._running is True
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
            mock_logger.log_info.assert_called_once_with("健康检查调度器已启动")

    def test_start_scheduler_already_running(self, scheduler, mock_logger):
        """测试调度器已在运行时启动"""
        scheduler._running = True

        result = scheduler.start_scheduler()

        assert result is True
        mock_logger.log_warning.assert_called_once_with("调度器已在运行")

    def test_stop_scheduler_success(self, scheduler, mock_logger):
        """测试成功停止调度器"""
        scheduler._running = True
        scheduler._scheduler_thread = Mock()

        result = scheduler.stop_scheduler()

        assert result is True
        assert scheduler._running is False
        scheduler._scheduler_thread.join.assert_called_once()
        mock_logger.log_info.assert_called_once_with("健康检查调度器已停止")

    def test_stop_scheduler_not_running(self, scheduler):
        """测试调度器未运行时停止"""
        scheduler._running = False

        result = scheduler.stop_scheduler()

        assert result is True

    def test_schedule_check_success(self, scheduler, mock_executor, mock_logger):
        """测试成功调度检查"""
        check = Mock(spec=HealthCheck)
        check.name = "test_check"
        check.enabled = True

        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            result = scheduler.schedule_check(check)

            assert result is None  # schedule_check returns None
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    def test_schedule_check_disabled_check(self, scheduler, mock_executor):
        """测试禁用检查的调度"""
        check = Mock(spec=HealthCheck)
        check.name = "test_check"
        check.enabled = False

        result = scheduler.schedule_check(check)

        assert result is None
        # Should not create thread for disabled check


    def test_get_scheduler_status(self, scheduler):
        """测试获取调度器状态"""
        scheduler._running = True
        mock_thread1 = Mock()
        mock_thread1.is_alive.return_value = True
        mock_thread2 = Mock()
        mock_thread2.is_alive.return_value = False
        scheduler._check_threads = {"check1": mock_thread1, "check2": mock_thread2}
        scheduler._last_execution = {"check1": 1234567890.0}

        status = scheduler.get_scheduler_status()

        assert status["running"] is True
        assert status["active_threads"] == 1  # Only check1 is alive
        assert "last_executions" in status
        assert status["last_executions"]["check1"] == 1234567890.0


    def test_scheduler_loop_basic(self, scheduler):
        """测试调度器循环基本功能"""
        scheduler._running = True

        with patch('time.sleep') as mock_sleep:
            # Run scheduler loop briefly
            def stop_after_iterations():
                count = 0
                while scheduler._running and count < 1:
                    scheduler._scheduler_loop()
                    count += 1
                scheduler._running = False

            stop_thread = threading.Thread(target=stop_after_iterations, daemon=True)
            stop_thread.start()
            stop_thread.join(timeout=1)

            mock_sleep.assert_called()


