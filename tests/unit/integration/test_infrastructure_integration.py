"""基础设施层集成测试"""
import pytest
import time
from unittest import mock
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.error.error_handler import ErrorHandler
from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
from src.infrastructure.resource.resource_manager import ResourceManager

class TestIntegration:
    def test_config_error_handling_integration(self, tmp_path):
        """测试配置管理与错误处理集成"""
        # 创建配置
        config_file = tmp_path / "test.yaml"
        config_file.write_text("key: value")

        # 初始化组件
        config = ConfigManager(config_dir=str(tmp_path))
        error_handler = ErrorHandler()

        # 注册配置错误处理器
        def handle_config_error(e):
            return "default_value"

        error_handler.register_handler(KeyError, handle_config_error)

        # 测试获取不存在的配置项
        result = error_handler.safe_execute(config.get, "nonexistent.key")
        assert result == "default_value"

    def test_resource_monitoring_integration(self, caplog):
        """测试资源监控与错误处理集成"""
        # 初始化组件
        resource_manager = ResourceManager(cpu_threshold=10.0)  # 设置低阈值便于测试
        app_monitor = ApplicationMonitor()
        error_handler = ErrorHandler()

        # 模拟高CPU使用率
        with mock.patch('psutil.cpu_percent', return_value=95.0):
            resource_manager.start_monitoring()
            time.sleep(0.2)
            resource_manager.stop_monitoring()

        # 验证错误处理
        errors = error_handler.get_error_history()
        assert any("CPU usage exceeds threshold" in e['message'] for e in errors)

        # 验证监控数据
        metrics = app_monitor.get_custom_metrics()
        assert len(metrics) > 0

    def test_monitoring_error_handling(self):
        """测试监控与错误处理集成"""
        app_monitor = ApplicationMonitor()
        error_handler = ErrorHandler()

        @app_monitor.monitor_function()
        @error_handler.safe_execute
        def problematic_function():
            raise ValueError("Simulated error")

        # 执行会出错的函数
        result = problematic_function()
        assert result is None

        # 验证错误记录
        errors = error_handler.get_error_history()
        assert len(errors) == 1
        assert "Simulated error" in errors[0]['message']

        # 验证监控记录
        metrics = app_monitor.get_function_metrics()
        assert len(metrics) == 1
        assert metrics[0]['success'] is False
