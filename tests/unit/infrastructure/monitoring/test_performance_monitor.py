import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.monitoring.performance_monitor import PerformanceMonitor
from src.infrastructure.error.exceptions import PerformanceThresholdExceeded

# 统一mock prometheus_client的指标对象
@pytest.fixture(autouse=True)
def mock_prometheus():
    with patch('prometheus_client.Counter', MagicMock()), \
         patch('prometheus_client.Gauge', MagicMock()), \
         patch('prometheus_client.Histogram', MagicMock()):
        yield

@pytest.fixture
def mock_config_manager():
    """创建mock的ConfigManager"""
    mock_cm = MagicMock()
    mock_cm.get_config.return_value = {
        'interval': 5,
        'storage': ['file'],
        'prometheus': {'endpoint': 'http://localhost:9090'},
        'influxdb': {'database': 'metrics'},
        'file': {'path': 'logs/performance.log'}
    }
    return mock_cm

@pytest.fixture
def performance_monitor(mock_config_manager):
    """创建性能监控器实例，使用mock的ConfigManager"""
    monitor = PerformanceMonitor(
        config={}, 
        config_manager=mock_config_manager
    )
    return monitor

def test_start_stop_monitoring(performance_monitor):
    """测试启动和停止监控"""
    # 启动监控
    performance_monitor.start()

    # 验证监控已启动
    assert performance_monitor.running

    # 停止监控
    performance_monitor.stop()

    # 验证监控已停止
    assert not performance_monitor.running

def test_record_metrics(performance_monitor):
    """测试记录性能指标"""
    # 记录指标
    from src.infrastructure.monitoring.performance_monitor import PerformanceMetric
    
    metric = PerformanceMetric(
        name="test.metric",
        value=75.5,
        timestamp=time.time(),
        tags={"test": "true"}
    )
    
    performance_monitor.record_metric(metric)

    # 验证指标被记录
    assert len(performance_monitor.metrics_buffer) == 1
    assert performance_monitor.metrics_buffer[0].name == "test.metric"
    assert performance_monitor.metrics_buffer[0].value == 75.5

def test_get_metrics(performance_monitor):
    """测试获取性能指标"""
    query = {"name": "system.cpu.usage"}
    metrics = performance_monitor.get_metrics(query)
    
    # 验证返回的指标格式
    assert isinstance(metrics, list)
    assert len(metrics) > 0
    assert "name" in metrics[0]
    assert "value" in metrics[0]
    assert "timestamp" in metrics[0]

def test_get_performance_report(performance_monitor):
    """测试获取性能报告"""
    report = performance_monitor.get_performance_report("1h")
    
    # 验证报告结构
    assert "cpu_usage" in report
    assert "memory_usage" in report
    assert "disk_io" in report
    assert "network_io" in report
    assert "service_metrics" in report

def test_check_health_status(performance_monitor):
    """测试健康状态检查"""
    health = performance_monitor.check_health_status()
    
    # 验证健康状态结构
    assert "status" in health
    assert "indicators" in health
    assert "anomalies" in health

def test_create_alert_rule(performance_monitor):
    """测试创建告警规则"""
    condition = "cpu_usage > 90"
    action = "send_alert"
    
    result = performance_monitor.create_alert_rule(condition, action)
    assert result is True

def test_track_service_metrics(performance_monitor):
    """测试跟踪服务指标"""
    service_name = "trading_service"
    metrics = {
        "latency": 12.5,
        "throughput": 1500,
        "error_rate": 0.01
    }
    
    performance_monitor.track_service_metrics(service_name, metrics)
    
    # 验证指标被记录
    assert len(performance_monitor.metrics_buffer) == 3
    metric_names = [m.name for m in performance_monitor.metrics_buffer]
    assert "service.trading_service.latency" in metric_names
    assert "service.trading_service.throughput" in metric_names
    assert "service.trading_service.error_rate" in metric_names
