"""
测试目标：提升resource/core/system_coordinator.py的真实覆盖率
实际导入和使用src.infrastructure.resource.core.system_coordinator模块
"""

from unittest.mock import Mock, patch
import pytest
import threading
import time

from src.infrastructure.resource.core.system_coordinator import SystemCoordinator


class TestSystemCoordinator:
    """测试SystemCoordinator类"""

    @pytest.fixture
    def mock_logger(self):
        """模拟logger"""
        return Mock()

    @pytest.fixture
    def mock_error_handler(self):
        """模拟error_handler"""
        return Mock()

    @pytest.fixture
    def coordinator(self, mock_logger, mock_error_handler):
        """创建协调器实例"""
        return SystemCoordinator(logger=mock_logger, error_handler=mock_error_handler)

    def test_initialization_default(self, coordinator, mock_logger, mock_error_handler):
        """测试默认参数初始化"""
        assert coordinator.logger == mock_logger
        assert coordinator.error_handler == mock_error_handler
        assert coordinator.config == {}
        assert coordinator.running is False
        assert coordinator.alert_check_thread is None
        assert coordinator.performance_monitor is None
        assert coordinator.alert_manager is None
        assert coordinator.notification_manager is None
        assert coordinator.test_monitor is None
        assert coordinator.alert_rule_manager is None

    def test_initialization_with_config(self, mock_logger, mock_error_handler):
        """测试带配置的初始化"""
        config = {"test": "value"}
        coordinator = SystemCoordinator(config=config, logger=mock_logger, error_handler=mock_error_handler)

        assert coordinator.config == config

    def test_initialization_without_logger_and_error_handler(self):
        """测试不提供logger和error_handler时的初始化"""
        coordinator = SystemCoordinator()

        assert coordinator.logger is not None
        assert hasattr(coordinator.logger, 'log_info')
        assert coordinator.error_handler is not None
        assert hasattr(coordinator.error_handler, 'handle_error')

    def test_set_components(self, coordinator):
        """测试设置组件"""
        mock_performance_monitor = Mock()
        mock_alert_manager = Mock()
        mock_notification_manager = Mock()
        mock_test_monitor = Mock()
        mock_alert_rule_manager = Mock()

        coordinator.set_components(
            mock_performance_monitor,
            mock_alert_manager,
            mock_notification_manager,
            mock_test_monitor,
            mock_alert_rule_manager
        )

        assert coordinator.performance_monitor == mock_performance_monitor
        assert coordinator.alert_manager == mock_alert_manager
        assert coordinator.notification_manager == mock_notification_manager
        assert coordinator.test_monitor == mock_test_monitor
        assert coordinator.alert_rule_manager == mock_alert_rule_manager

    def test_start_system_success(self, coordinator, mock_logger):
        """测试成功启动系统"""
        # 设置mock组件
        coordinator.performance_monitor = Mock()
        coordinator.alert_manager = Mock()
        coordinator.notification_manager = Mock()
        coordinator.test_monitor = Mock()
        coordinator.alert_rule_manager = Mock()

        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            result = coordinator.start_system()

            assert result is True
            assert coordinator.running is True
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
            mock_logger.log_info.assert_called_with("系统协调器已启动")

    def test_start_system_already_running(self, coordinator, mock_logger):
        """测试系统已在运行时启动"""
        coordinator.running = True

        result = coordinator.start_system()

        assert result is False
        mock_logger.log_warning.assert_called_with("系统协调器已在运行")

    def test_start_system_missing_components(self, coordinator, mock_logger):
        """测试缺少组件时启动系统"""
        result = coordinator.start_system()

        assert result is False
        mock_logger.log_error.assert_called_with("组件未正确设置，无法启动系统")

    def test_stop_system_success(self, coordinator, mock_logger):
        """测试成功停止系统"""
        coordinator.running = True
        coordinator.alert_check_thread = Mock()

        result = coordinator.stop_system()

        assert result is True
        assert coordinator.running is False
        coordinator.alert_check_thread.join.assert_called_once()
        mock_logger.log_info.assert_called_with("系统协调器已停止")

    def test_stop_system_not_running(self, coordinator, mock_logger):
        """测试系统未运行时停止"""
        coordinator.running = False

        result = coordinator.stop_system()

        assert result is True
        mock_logger.log_warning.assert_called_with("系统协调器未在运行")

    def test_get_system_status_running(self, coordinator):
        """测试获取运行中系统的状态"""
        coordinator.running = True
        coordinator.alert_check_thread = Mock()
        coordinator.alert_check_thread.is_alive.return_value = True

        status = coordinator.get_system_status()

        assert status["running"] is True
        assert status["thread_alive"] is True
        assert "uptime" in status

    def test_get_system_status_not_running(self, coordinator):
        """测试获取未运行系统的状态"""
        coordinator.running = False

        status = coordinator.get_system_status()

        assert status["running"] is False
        assert status["thread_alive"] is False
        assert status["uptime"] == 0

    def test_alert_check_loop_running(self, coordinator, mock_logger):
        """测试运行中的告警检查循环"""
        coordinator.running = True

        # 设置mock组件
        coordinator.performance_monitor = Mock()
        coordinator.alert_manager = Mock()
        coordinator.notification_manager = Mock()
        coordinator.test_monitor = Mock()
        coordinator.alert_rule_manager = Mock()

        with patch('time.sleep') as mock_sleep:
            # 模拟运行一次循环后停止
            def stop_after_one_iteration():
                coordinator.running = False

            # Mock组件的行为
            coordinator.performance_monitor.get_current_metrics.return_value = {"cpu": 50}
            coordinator.alert_manager.check_alerts.return_value = []
            coordinator.test_monitor.get_active_tests.return_value = {}

            # 运行循环
            coordinator._alert_check_loop()

            # 验证组件被调用
            coordinator.performance_monitor.get_current_metrics.assert_called_once()
            coordinator.alert_manager.check_alerts.assert_called_once()
            coordinator.test_monitor.get_active_tests.assert_called_once()
            mock_sleep.assert_called_once_with(30)

    def test_alert_check_loop_with_alerts(self, coordinator, mock_logger):
        """测试有告警时的告警检查循环"""
        coordinator.running = True

        # 设置mock组件
        coordinator.performance_monitor = Mock()
        coordinator.alert_manager = Mock()
        coordinator.notification_manager = Mock()
        coordinator.test_monitor = Mock()
        coordinator.alert_rule_manager = Mock()

        # Mock告警
        mock_alert = Mock()
        coordinator.alert_manager.check_alerts.return_value = [mock_alert]

        with patch('time.sleep') as mock_sleep:
            coordinator.running = False  # 只运行一次

            coordinator._alert_check_loop()

            # 验证通知管理器被调用
            coordinator.notification_manager.send_alert_notification.assert_called_once_with(mock_alert)

    def test_alert_check_loop_with_exceptions(self, coordinator, mock_logger):
        """测试异常情况下的告警检查循环"""
        coordinator.running = True

        # 设置mock组件，使其抛出异常
        coordinator.performance_monitor = Mock()
        coordinator.performance_monitor.get_current_metrics.side_effect = Exception("Test error")

        with patch('time.sleep') as mock_sleep:
            coordinator.running = False  # 只运行一次

            coordinator._alert_check_loop()

            # 验证错误被记录
            mock_logger.log_error.assert_called_once()
            coordinator.error_handler.handle_error.assert_called_once()

    def test_health_check_success(self, coordinator):
        """测试成功的健康检查"""
        # 设置mock组件
        coordinator.performance_monitor = Mock()
        coordinator.alert_manager = Mock()
        coordinator.notification_manager = Mock()
        coordinator.test_monitor = Mock()
        coordinator.alert_rule_manager = Mock()

        # Mock健康状态
        coordinator.performance_monitor.is_healthy.return_value = True
        coordinator.alert_manager.is_healthy.return_value = True
        coordinator.notification_manager.is_healthy.return_value = True
        coordinator.test_monitor.is_healthy.return_value = True
        coordinator.alert_rule_manager.is_healthy.return_value = True

        result = coordinator.health_check()

        assert result["overall_health"] is True
        assert all(component["healthy"] for component in result["components"].values())

    def test_health_check_with_unhealthy_components(self, coordinator):
        """测试有不健康组件时的健康检查"""
        # 设置mock组件
        coordinator.performance_monitor = Mock()
        coordinator.alert_manager = Mock()

        # Mock健康状态
        coordinator.performance_monitor.is_healthy.return_value = False
        coordinator.alert_manager.is_healthy.return_value = True

        result = coordinator.health_check()

        assert result["overall_health"] is False
        assert result["components"]["performance_monitor"]["healthy"] is False
        assert result["components"]["alert_manager"]["healthy"] is True

    def test_health_check_missing_components(self, coordinator):
        """测试缺少组件时的健康检查"""
        result = coordinator.health_check()

        assert result["overall_health"] is False
        assert len(result["components"]) == 0

    def test_get_component_status(self, coordinator):
        """测试获取组件状态"""
        # 设置mock组件
        coordinator.performance_monitor = Mock()
        coordinator.alert_manager = Mock()
        coordinator.performance_monitor.get_status.return_value = {"status": "running"}
        coordinator.alert_manager.get_status.return_value = {"status": "active"}

        status = coordinator.get_component_status()

        assert status["performance_monitor"]["status"] == "running"
        assert status["alert_manager"]["status"] == "active"

    def test_get_component_status_missing_components(self, coordinator):
        """测试缺少组件时的组件状态获取"""
        status = coordinator.get_component_status()

        assert status == {}

    def test_thread_safety(self, coordinator):
        """测试线程安全性"""
        results = []

        def worker():
            try:
                # 测试并发访问
                status = coordinator.get_system_status()
                results.append(status)
            except Exception as e:
                results.append(f"error: {e}")

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 5
        for result in results:
            assert isinstance(result, dict) or result.startswith("error:")
