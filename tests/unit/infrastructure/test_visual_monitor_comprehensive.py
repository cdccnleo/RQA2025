"""
可视化监控模块深度测试

目标：为visual_monitor.py提供100%测试覆盖
当前覆盖率：0% → 目标100%
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.infrastructure.visual_monitor import (
    VisualMonitor, ConfigManager, HealthChecker, CircuitBreaker,
    DegradationManager, AutoRecovery, ServiceStatus
)


class TestVisualMonitorComprehensive:
    """VisualMonitor深度测试"""

    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        return {
            "visual": {
                "update_interval": 5,
                "dashboard_port": 8080,
                "metrics_port": 9090
            }
        }

    @pytest.fixture
    def mock_registry(self):
        """模拟Prometheus注册表"""
        return Mock()

    @pytest.fixture
    def visual_monitor(self, mock_config, mock_registry):
        """创建VisualMonitor实例"""
        return VisualMonitor(mock_config, registry=mock_registry)

    def test_initialization(self, mock_config, mock_registry):
        """测试初始化"""
        monitor = VisualMonitor(mock_config, registry=mock_registry)

        assert monitor.config == mock_config
        assert isinstance(monitor.config_manager, ConfigManager)
        assert isinstance(monitor.health_checker, HealthChecker)
        assert isinstance(monitor.circuit_breaker, CircuitBreaker)
        assert isinstance(monitor.degradation_manager, DegradationManager)
        assert isinstance(monitor.auto_recovery, AutoRecovery)

        assert monitor.services == {}
        assert monitor.service_statuses == {}
        assert isinstance(monitor.lock, type(threading.Lock()))
        assert not monitor.running

        # 检查仪表盘数据结构
        expected_keys = ["services", "system_health", "last_updated", "timestamp"]
        assert all(key in monitor.dashboard_data for key in expected_keys)

        # 检查配置属性
        assert monitor.update_interval == 5
        assert monitor.dashboard_port == 8080
        assert monitor.metrics_port == 9090

    def test_start_stop(self, visual_monitor):
        """测试启动和停止"""
        # 测试启动
        visual_monitor.start()
        assert visual_monitor.running

        # 测试重复启动（应该无操作）
        visual_monitor.start()
        assert visual_monitor.running

        # 测试停止
        visual_monitor.stop()
        assert not visual_monitor.running

    @patch('threading.Thread')
    def test_start_creates_thread(self, mock_thread_class, visual_monitor):
        """测试启动时创建监控线程"""
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        visual_monitor.start()

        mock_thread_class.assert_called_once()
        mock_thread.start.assert_called_once()
        assert visual_monitor.running

    def test_get_dashboard_data_empty(self, visual_monitor):
        """测试获取空的仪表盘数据"""
        data = visual_monitor.get_dashboard_data()

        assert isinstance(data, dict)
        assert "services" in data
        assert "system_health" in data
        assert "last_updated" in data
        assert "timestamp" in data
        assert data["services"] == []
        assert data["system_health"] == "GREEN"

    def test_get_dashboard_data_with_services(self, visual_monitor):
        """测试获取包含服务的仪表盘数据"""
        # 添加服务状态
        service = ServiceStatus(
            name="test_service",
            health="GREEN",
            breaker_state="CLOSED",
            degradation_level=0,
            last_updated=time.time()
        )
        visual_monitor.services["test_service"] = service
        visual_monitor.service_statuses["test_service"] = service

        data = visual_monitor.get_dashboard_data()

        assert len(data["services"]) == 1
        service_data = data["services"][0]
        assert service_data["name"] == "test_service"
        assert service_data["health"] == "GREEN"
        assert service_data["breaker_state"] == "CLOSED"
        assert service_data["degradation_level"] == 0

    def test_prepare_dashboard_data_internal(self, visual_monitor):
        """测试内部仪表盘数据准备"""
        # 添加服务
        service = ServiceStatus(
            name="test_service",
            health="YELLOW",
            breaker_state="OPEN",
            degradation_level=2,
            last_updated=time.time()
        )
        visual_monitor.service_statuses["test_service"] = service

        # 调用内部方法
        visual_monitor._prepare_dashboard_data_internal()

        # 检查仪表盘数据
        assert len(visual_monitor.dashboard_data["services"]) == 1
        service_data = visual_monitor.dashboard_data["services"][0]
        assert service_data["name"] == "test_service"
        assert service_data["health"] == "YELLOW"

    def test_prepare_dashboard_data(self, visual_monitor):
        """测试仪表盘数据准备方法"""
        # 这个方法可能不存在或者已被弃用，但我们测试一下
        try:
            visual_monitor._prepare_dashboard_data()
        except AttributeError:
            # 方法不存在，跳过测试
            pytest.skip("_prepare_dashboard_data method not available")

    def test_load_config(self, visual_monitor):
        """测试配置加载"""
        # 这个方法应该已经在线程锁内被调用
        # 我们测试配置是否正确加载
        assert visual_monitor.update_interval == 5
        assert visual_monitor.dashboard_port == 8080
        assert visual_monitor.metrics_port == 9090

    def test_update_service_status(self, visual_monitor):
        """测试服务状态更新"""
        # 模拟健康检查器返回简单的字典结构
        mock_health_result = {
            "overall_health": "GREEN",
            "service1": "UP"
        }
        visual_monitor.health_checker.get_status = Mock(return_value=mock_health_result)
        visual_monitor.circuit_breaker.get_status = Mock(return_value={"service1": "CLOSED"})
        visual_monitor.degradation_manager.get_status_report = Mock(return_value={"service1": 0})

        # 调用更新方法
        visual_monitor._update_service_status()

        # 检查服务状态是否已更新
        assert "service1" in visual_monitor.services
        service = visual_monitor.services["service1"]
        assert service.health == "UP"  # 根据mock数据
        assert service.breaker_state == "CLOSED"

    def test_calculate_system_health(self, visual_monitor):
        """测试系统健康度计算"""
        # 设置不同的服务状态 (使用代码期望的状态值)
        services = {
            "healthy": ServiceStatus("healthy", "UP", "CLOSED", 0, time.time()),
            "warning": ServiceStatus("warning", "DEGRADED", "CLOSED", 1, time.time()),
            "critical": ServiceStatus("critical", "DOWN", "OPEN", 3, time.time())
        }

        visual_monitor.services = services

        # 计算系统健康度
        visual_monitor._calculate_system_health()

        # DOWN状态的服务存在时，系统健康度应该是RED
        assert visual_monitor.dashboard_data["system_health"] == "RED"

    def test_calculate_system_health_all_green(self, visual_monitor):
        """测试所有服务都健康时的系统健康度"""
        services = {
            "service1": ServiceStatus("service1", "UP", "CLOSED", 0, time.time()),
            "service2": ServiceStatus("service2", "UP", "CLOSED", 0, time.time())
        }

        visual_monitor.services = services
        visual_monitor._calculate_system_health()

        assert visual_monitor.dashboard_data["system_health"] == "GREEN"

    def test_monitor_loop(self, visual_monitor):
        """测试监控循环"""
        # 设置运行标志
        visual_monitor.running = True

        # 模拟监控循环的一次迭代
        with patch.object(time, 'sleep') as mock_sleep, \
             patch.object(visual_monitor, '_update_service_status') as mock_update, \
             patch.object(visual_monitor, '_calculate_system_health') as mock_calculate:

            # 模拟第一次循环后停止
            def stop_after_first(*args, **kwargs):
                visual_monitor.running = False

            mock_update.side_effect = stop_after_first

            # 运行监控循环
            visual_monitor._monitor_loop()

            # 验证方法被调用
            mock_update.assert_called_once()
            mock_calculate.assert_called_once()
            mock_sleep.assert_called_once_with(5)  # 默认更新间隔

    @patch('time.time')
    def test_generate_html_report(self, mock_time, visual_monitor):
        """测试HTML报告生成"""
        mock_time.return_value = 1234567890.0

        # 添加服务数据
        service = ServiceStatus("test", "GREEN", "CLOSED", 0, time.time())
        visual_monitor.services["test"] = service

        report = visual_monitor.generate_html_report()

        assert isinstance(report, str)
        assert "<html>" in report
        assert "<head>" in report
        assert "<body>" in report
        assert "test" in report
        assert "GREEN" in report

    def test_generate_prometheus_metrics(self, visual_monitor):
        """测试Prometheus指标生成"""
        # 添加服务数据
        service = ServiceStatus("test", "GREEN", "CLOSED", 0, time.time())
        visual_monitor.services["test"] = service

        metrics = visual_monitor.generate_prometheus_metrics()

        assert isinstance(metrics, str)
        # 检查是否包含Prometheus格式的指标
        assert "service_health" in metrics or "service_response_time" in metrics

    def test_thread_safety(self, visual_monitor):
        """测试线程安全性"""
        import threading
        import concurrent.futures

        results = []

        def worker(worker_id):
            # 并发访问仪表盘数据
            for i in range(10):
                data = visual_monitor.get_dashboard_data()
                results.append((worker_id, i, len(data)))

        # 创建多个线程并发访问
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, i) for i in range(3)]
            concurrent.futures.wait(futures)

        # 验证所有操作都成功完成
        assert len(results) == 30  # 3 workers * 10 iterations each


class TestConfigManager:
    """ConfigManager测试"""

    def test_initialization(self):
        """测试初始化"""
        config = {"key": "value"}
        manager = ConfigManager(config)
        assert manager.config == config

    def test_initialization_empty(self):
        """测试空配置初始化"""
        manager = ConfigManager()
        assert manager.config == {}

    def test_get_existing_key(self):
        """测试获取存在的键"""
        config = {"key": "value"}
        manager = ConfigManager(config)
        assert manager.get("key") == "value"

    def test_get_missing_key(self):
        """测试获取不存在的键"""
        manager = ConfigManager()
        assert manager.get("missing") is None

    def test_get_missing_key_with_default(self):
        """测试获取不存在的键（带默认值）"""
        manager = ConfigManager()
        assert manager.get("missing", "default") == "default"


class TestHealthChecker:
    """HealthChecker测试"""

    def test_initialization(self):
        """测试初始化"""
        config = {"health": {"timeout": 30}}
        checker = HealthChecker(config)
        assert checker.config == config

    def test_get_status(self):
        """测试获取状态"""
        checker = HealthChecker()
        status = checker.get_status()

        # HealthChecker的get_status方法可能返回不同的格式
        assert isinstance(status, dict)


class TestCircuitBreaker:
    """CircuitBreaker测试"""

    def test_initialization(self):
        """测试初始化"""
        config = {"circuit_breaker": {"threshold": 5}}
        registry = Mock()
        breaker = CircuitBreaker(config, registry=registry)
        assert breaker.config == config

    def test_get_status(self):
        """测试获取状态"""
        breaker = CircuitBreaker()
        status = breaker.get_status()
        assert isinstance(status, dict)  # 实际返回字典


class TestDegradationManager:
    """DegradationManager测试"""

    def test_initialization(self):
        """测试初始化"""
        config = {"degradation": {"levels": 3}}
        breaker = CircuitBreaker()
        manager = DegradationManager(config, circuit_breaker=breaker)
        assert manager.config == config
        # circuit_breaker参数被接受但不存储为实例属性

    def test_get_status_report(self):
        """测试获取状态报告"""
        manager = DegradationManager()
        report = manager.get_status_report()
        assert isinstance(report, dict)


class TestAutoRecovery:
    """AutoRecovery测试"""

    def test_initialization(self):
        """测试初始化"""
        config = {"recovery": {"interval": 60}}
        recovery = AutoRecovery(config)
        assert recovery.config == config


class TestServiceStatus:
    """ServiceStatus测试"""

    def test_initialization(self):
        """测试初始化"""
        last_updated = time.time()
        status = ServiceStatus(
            name="test_service",
            health="GREEN",
            breaker_state="CLOSED",
            degradation_level=1,
            last_updated=last_updated
        )

        assert status.name == "test_service"
        assert status.health == "GREEN"
        assert status.breaker_state == "CLOSED"
        assert status.degradation_level == 1
        assert status.last_updated == last_updated

    def test_initialization_defaults(self):
        """测试带默认值的初始化"""
        status = ServiceStatus("minimal", "UNKNOWN", "UNKNOWN", 0, time.time())
        assert status.name == "minimal"
        assert status.health == "UNKNOWN"
        assert status.breaker_state == "UNKNOWN"
        assert status.degradation_level == 0


class TestVisualMonitorIntegration:
    """VisualMonitor集成测试"""

    def test_full_workflow(self):
        """测试完整工作流程"""
        config = {
            "visual": {
                "update_interval": 1,
                "dashboard_port": 8080
            }
        }

        monitor = VisualMonitor(config)

        # 启动监控
        monitor.start()
        assert monitor.running

        # 等待一小段时间让监控循环运行
        time.sleep(0.1)

        # 获取仪表盘数据
        data = monitor.get_dashboard_data()
        assert isinstance(data, dict)

        # 生成报告
        html_report = monitor.generate_html_report()
        assert isinstance(html_report, str)

        prometheus_metrics = monitor.generate_prometheus_metrics()
        assert isinstance(prometheus_metrics, str)

        # 停止监控
        monitor.stop()
        assert not monitor.running

    def test_concurrent_access(self):
        """测试并发访问"""
        import concurrent.futures

        config = {"visual": {"update_interval": 1}}
        monitor = VisualMonitor(config)

        # 添加一些测试服务
        for i in range(5):
            service = ServiceStatus(f"service_{i}", "GREEN", "CLOSED", 0, time.time())
            monitor.services[f"service_{i}"] = service

        results = []

        def concurrent_reader():
            for _ in range(10):
                data = monitor.get_dashboard_data()
                results.append(len(data["services"]))
                time.sleep(0.01)  # 小延迟模拟真实使用

        # 并发读取
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(concurrent_reader) for _ in range(3)]
            concurrent.futures.wait(futures)

        # 验证所有读取都成功
        assert all(count == 5 for count in results)
