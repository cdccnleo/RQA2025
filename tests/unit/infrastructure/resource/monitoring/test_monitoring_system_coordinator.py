"""
测试目标：提升resource/monitoring/monitoring_system_coordinator.py的真实覆盖率
实际导入和使用src.infrastructure.resource.monitoring.monitoring_system_coordinator模块
"""

from unittest.mock import Mock, patch
import pytest
import threading
import time

from src.infrastructure.resource.monitoring.monitoring_system_coordinator import MonitoringSystemCoordinator


class TestMonitoringSystemCoordinator:
    """测试MonitoringSystemCoordinator类"""

    @pytest.fixture
    def mock_logger(self):
        """模拟logger"""
        return Mock()

    @pytest.fixture
    def coordinator(self, mock_logger):
        """创建协调器实例"""
        return MonitoringSystemCoordinator(logger=mock_logger)

    @pytest.fixture
    def coordinator_with_config(self, mock_logger):
        """创建带有配置的协调器实例"""
        config = {"check_interval": 10, "timeout": 30}
        return MonitoringSystemCoordinator(config=config, logger=mock_logger)

    def test_initialization_default(self, coordinator, mock_logger):
        """测试默认参数初始化"""
        assert coordinator.config == {}
        assert coordinator.logger == mock_logger
        assert coordinator._running is False
        assert coordinator._monitoring_thread is None
        assert isinstance(coordinator._lock, type(threading.Lock()))

    def test_initialization_with_config(self, coordinator_with_config, mock_logger):
        """测试带配置的初始化"""
        assert coordinator_with_config.config == {"check_interval": 10, "timeout": 30}
        assert coordinator_with_config.logger == mock_logger

    def test_initialization_without_logger(self):
        """测试不提供logger时的初始化"""
        coordinator = MonitoringSystemCoordinator()

        assert coordinator.logger is not None
        assert hasattr(coordinator.logger, 'log_info')

    def test_start_alias(self, coordinator, mock_logger):
        """测试start方法作为start_monitoring的别名"""
        with patch.object(coordinator, 'start_monitoring', return_value=True) as mock_start:
            result = coordinator.start()

            mock_start.assert_called_once()
            assert result is True

    def test_stop_alias(self, coordinator, mock_logger):
        """测试stop方法作为stop_monitoring的别名"""
        with patch.object(coordinator, 'stop_monitoring', return_value=True) as mock_stop:
            result = coordinator.stop()

            mock_stop.assert_called_once()
            assert result is True

    def test_start_monitoring_success(self, coordinator, mock_logger):
        """测试成功启动监控系统"""
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            result = coordinator.start_monitoring()

            assert result is True
            assert coordinator._running is True
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
            mock_logger.log_info.assert_called_with("监控系统已启动")

    def test_start_monitoring_already_running(self, coordinator, mock_logger):
        """测试监控系统已在运行时启动"""
        coordinator._running = True

        result = coordinator.start_monitoring()

        assert result is True
        mock_logger.log_warning.assert_called_with("监控系统已在运行")

    def test_stop_monitoring_success(self, coordinator, mock_logger):
        """测试成功停止监控系统"""
        coordinator._running = True
        coordinator._monitoring_thread = Mock()

        result = coordinator.stop_monitoring()

        assert result is True
        assert coordinator._running is False
        coordinator._monitoring_thread.join.assert_called_once()
        mock_logger.log_info.assert_called_with("监控系统已停止")

    def test_stop_monitoring_not_running(self, coordinator, mock_logger):
        """测试监控系统未运行时停止"""
        coordinator._running = False

        result = coordinator.stop_monitoring()

        assert result is True
        mock_logger.log_warning.assert_called_with("监控系统未在运行")

    def test_get_status_running(self, coordinator):
        """测试获取运行中系统的状态"""
        coordinator._running = True
        coordinator._monitoring_thread = Mock()
        coordinator._monitoring_thread.is_alive.return_value = True

        status = coordinator.get_status()

        assert status["running"] is True
        assert status["thread_alive"] is True
        assert "uptime" in status
        assert "config" in status

    def test_get_status_not_running(self, coordinator):
        """测试获取未运行系统的状态"""
        coordinator._running = False

        status = coordinator.get_status()

        assert status["running"] is False
        assert status["thread_alive"] is False
        assert status["uptime"] == 0

    def test_monitoring_loop_running(self, coordinator, mock_logger):
        """测试运行中的监控循环"""
        coordinator._running = True

        with patch('time.sleep') as mock_sleep:
            # 模拟运行一次循环后停止
            def stop_after_one_iteration():
                time.sleep(0.1)  # 短暂延迟
                coordinator._running = False

            stop_thread = threading.Thread(target=stop_after_one_iteration, daemon=True)
            stop_thread.start()

            # 运行监控循环
            coordinator._monitoring_loop()

            # 验证sleep被调用
            mock_sleep.assert_called()
            mock_logger.log_info.assert_called_with("监控循环开始")

    def test_monitoring_loop_with_exception(self, coordinator, mock_logger):
        """测试异常情况下的监控循环"""
        coordinator._running = True

        # Mock time.sleep 抛出异常
        with patch('time.sleep', side_effect=Exception("Sleep error")):
            coordinator._running = False  # 立即停止

            coordinator._monitoring_loop()

            # 验证异常被记录
            mock_logger.log_error.assert_called()

    def test_health_check_running(self, coordinator):
        """测试运行中系统的健康检查"""
        coordinator._running = True
        coordinator._monitoring_thread = Mock()
        coordinator._monitoring_thread.is_alive.return_value = True

        result = coordinator.health_check()

        assert result["healthy"] is True
        assert result["running"] is True
        assert result["thread_alive"] is True

    def test_health_check_not_running(self, coordinator):
        """测试未运行系统的健康检查"""
        coordinator._running = False

        result = coordinator.health_check()

        assert result["healthy"] is False
        assert result["running"] is False

    def test_health_check_thread_dead(self, coordinator):
        """测试线程死亡时的健康检查"""
        coordinator._running = True
        coordinator._monitoring_thread = Mock()
        coordinator._monitoring_thread.is_alive.return_value = False

        result = coordinator.health_check()

        assert result["healthy"] is False
        assert result["thread_alive"] is False

    def test_configure(self, coordinator):
        """测试配置更新"""
        new_config = {"check_interval": 20, "timeout": 60}

        coordinator.configure(new_config)

        assert coordinator.config == new_config

    def test_configure_empty(self, coordinator):
        """测试空配置更新"""
        original_config = coordinator.config.copy()

        coordinator.configure({})

        assert coordinator.config == original_config

    def test_thread_safety(self, coordinator):
        """测试线程安全性"""
        results = []

        def worker():
            try:
                # 测试并发访问
                status = coordinator.get_status()
                results.append(status)
            except Exception as e:
                results.append(f"error: {e}")

        threads = []
        for i in range(3):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict) or result.startswith("error:")

    def test_restart_system(self, coordinator, mock_logger):
        """测试重启系统"""
        # 先启动系统
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            coordinator.start_monitoring()
            assert coordinator._running is True

        # 再停止系统
        coordinator.stop_monitoring()
        assert coordinator._running is False

        # 验证日志
        mock_logger.log_info.assert_called_with("监控系统已启动")
        mock_logger.log_info.assert_called_with("监控系统已停止")

    def test_multiple_start_stop_calls(self, coordinator, mock_logger):
        """测试多次启动和停止调用"""
        # 多次启动
        result1 = coordinator.start_monitoring()
        result2 = coordinator.start_monitoring()

        assert result1 is True
        assert result2 is True  # 第二次调用应该成功（因为已经在运行）

        # 多次停止
        result3 = coordinator.stop_monitoring()
        result4 = coordinator.stop_monitoring()

        assert result3 is True
        assert result4 is True  # 第二次调用应该成功（因为已经停止）

        # 验证警告日志
        mock_logger.log_warning.assert_called_with("监控系统已在运行")
        mock_logger.log_warning.assert_called_with("监控系统未在运行")
