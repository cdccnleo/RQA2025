from unittest.mock import Mock, patch

from src.infrastructure.monitoring.services.unified_monitoring_service import (
    UnifiedMonitoring,
    get_unified_monitoring,
    create_monitoring_service,
)


@patch("src.infrastructure.monitoring.services.unified_monitoring_service.ContinuousMonitoringSystem")
def test_initialize_success(continuous_monitoring_cls):
    system_instance = Mock()
    continuous_monitoring_cls.return_value = system_instance

    monitoring = UnifiedMonitoring()

    assert monitoring.initialize() is True
    continuous_monitoring_cls.assert_called_once_with()
    assert monitoring._initialized is True
    assert monitoring._monitoring_system is system_instance


@patch("src.infrastructure.monitoring.services.unified_monitoring_service.ContinuousMonitoringSystem", side_effect=RuntimeError("boom"))
def test_initialize_failure(continuous_monitoring_cls):
    monitoring = UnifiedMonitoring()

    assert monitoring.initialize() is False
    assert monitoring._initialized is False
    assert monitoring._monitoring_system is None


def test_start_monitoring_requires_initialization():
    monitoring = UnifiedMonitoring()
    assert monitoring.start_monitoring() is False


def test_start_monitoring_success():
    monitoring = UnifiedMonitoring()
    monitoring._initialized = True
    monitoring._monitoring_system = Mock()

    assert monitoring.start_monitoring() is True
    monitoring._monitoring_system.start_monitoring.assert_called_once()


def test_start_monitoring_failure():
    monitoring = UnifiedMonitoring()
    monitoring._initialized = True
    failing_system = Mock()
    failing_system.start_monitoring.side_effect = RuntimeError("boom")
    monitoring._monitoring_system = failing_system

    assert monitoring.start_monitoring() is False


def test_stop_monitoring_without_system():
    monitoring = UnifiedMonitoring()
    assert monitoring.stop_monitoring() is True


def test_stop_monitoring_success():
    monitoring = UnifiedMonitoring()
    monitoring._monitoring_system = Mock()

    assert monitoring.stop_monitoring() is True
    monitoring._monitoring_system.stop_monitoring.assert_called_once()


def test_stop_monitoring_failure():
    monitoring = UnifiedMonitoring()
    failing_system = Mock()
    failing_system.stop_monitoring.side_effect = RuntimeError("boom")
    monitoring._monitoring_system = failing_system

    assert monitoring.stop_monitoring() is False


def test_get_monitoring_report_not_initialized():
    monitoring = UnifiedMonitoring()
    report = monitoring.get_monitoring_report()

    assert report["status"] == "error"
    assert "未初始化" in report["message"]


def test_get_monitoring_report_success():
    monitoring = UnifiedMonitoring()
    monitoring._initialized = True
    monitoring._monitoring_system = Mock()
    monitoring._monitoring_system.get_monitoring_report.return_value = {"status": "ok"}

    assert monitoring.get_monitoring_report() == {"status": "ok"}


def test_get_monitoring_report_failure():
    monitoring = UnifiedMonitoring()
    monitoring._initialized = True
    failing_system = Mock()
    failing_system.get_monitoring_report.side_effect = RuntimeError("boom")
    monitoring._monitoring_system = failing_system

    report = monitoring.get_monitoring_report()
    assert report["status"] == "error"
    assert "失败" in report["message"]


def test_status_and_health_check():
    monitoring = UnifiedMonitoring()
    monitoring._initialized = True
    monitoring._monitoring_system = Mock()
    monitoring._monitoring_system.monitoring_active = True

    status = monitoring.get_status()
    assert status["initialized"] is True
    assert status["running"] is True

    health = monitoring.health_check()
    assert health["healthy"] is True
    assert health["status"] == status


@patch.object(UnifiedMonitoring, "initialize", return_value=True)
def test_get_unified_monitoring_helper(mock_initialize):
    monitoring = get_unified_monitoring()
    assert isinstance(monitoring, UnifiedMonitoring)
    mock_initialize.assert_called_once_with()


@patch.object(UnifiedMonitoring, "initialize", return_value=True)
def test_create_monitoring_service_helper(mock_initialize):
    config = {"interval_seconds": 100}
    monitoring = create_monitoring_service(config)
    assert isinstance(monitoring, UnifiedMonitoring)
    mock_initialize.assert_called_once_with(config)


@patch("src.infrastructure.monitoring.services.unified_monitoring_service.ContinuousMonitoringSystem")
def test_initialize_with_config(continuous_monitoring_cls):
    """测试使用配置参数初始化"""
    system_instance = Mock()
    continuous_monitoring_cls.return_value = system_instance
    
    monitoring = UnifiedMonitoring()
    config = {"interval_seconds": 60}
    
    assert monitoring.initialize(config) is True
    continuous_monitoring_cls.assert_called_once_with()
    assert monitoring._initialized is True


def test_get_monitoring_report_system_none():
    """测试获取监控报告时系统为None的情况"""
    monitoring = UnifiedMonitoring()
    monitoring._initialized = True
    monitoring._monitoring_system = None
    
    report = monitoring.get_monitoring_report()
    assert report["status"] == "error"
    assert "未初始化" in report["message"]


def test_get_status_system_none():
    """测试获取状态时系统为None的情况"""
    monitoring = UnifiedMonitoring()
    monitoring._initialized = True
    monitoring._monitoring_system = None
    
    status = monitoring.get_status()
    assert status["initialized"] is True
    assert status["running"] is False


def test_get_status_not_initialized():
    """测试获取状态时未初始化的情况"""
    monitoring = UnifiedMonitoring()
    
    status = monitoring.get_status()
    assert status["initialized"] is False
    assert status["running"] is False


def test_health_check_not_healthy():
    """测试健康检查返回不健康状态"""
    monitoring = UnifiedMonitoring()
    monitoring._initialized = False
    
    health = monitoring.health_check()
    assert health["healthy"] is False
    assert health["service"] == "unified_monitoring"


def test_health_check_not_running():
    """测试健康检查时未运行的情况"""
    monitoring = UnifiedMonitoring()
    monitoring._initialized = True
    monitoring._monitoring_system = Mock()
    monitoring._monitoring_system.monitoring_active = False
    
    health = monitoring.health_check()
    assert health["healthy"] is False


def test_destructor_calls_stop():
    """测试析构函数调用stop_monitoring"""
    monitoring = UnifiedMonitoring()
    monitoring._monitoring_system = Mock()
    
    # 手动调用析构函数
    monitoring.__del__()
    
    monitoring._monitoring_system.stop_monitoring.assert_called_once()


def test_destructor_without_system():
    """测试析构函数在没有系统时不会出错"""
    monitoring = UnifiedMonitoring()
    monitoring._monitoring_system = None
    
    # 应该不会抛出异常
    monitoring.__del__()
