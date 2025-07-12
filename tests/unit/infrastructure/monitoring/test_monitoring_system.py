import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.monitoring.monitoring_system import (
    AlertLevel, Metric, Counter, Gauge, AlertChannel,
    DingTalkChannel, WeComChannel, MonitoringSystem
)

class TestAlertLevel:
    """AlertLevel枚举测试"""

    def test_alert_levels(self):
        """测试告警级别枚举值"""
        assert AlertLevel.INFO.value == 1
        assert AlertLevel.WARNING.value == 2
        assert AlertLevel.CRITICAL.value == 3

class TestMetric:
    """Metric基类测试"""

    @pytest.fixture
    def metric(self):
        """创建Metric实例"""
        return Metric("test_metric", "Test metric description", ["label1", "label2"])

    def test_metric_init(self, metric):
        """测试Metric初始化"""
        assert metric.name == "test_metric"
        assert metric.description == "Test metric description"
        assert metric.labels == ["label1", "label2"]

    def test_metric_without_labels(self):
        """测试无标签的Metric"""
        metric = Metric("simple_metric", "Simple metric")
        assert metric.labels == []

class TestCounter:
    """Counter类测试"""

    @pytest.fixture
    def counter(self):
        """创建Counter实例"""
        return Counter("test_counter", "Test counter", ["label1", "label2"])

    def test_counter_init(self, counter):
        """测试Counter初始化"""
        assert counter.name == "test_counter"
        assert counter.description == "Test counter"
        assert counter.labels == ["label1", "label2"]

    def test_counter_increment(self, counter):
        """测试计数器增加"""
        counter.inc()
        assert counter.get() == 1

        counter.inc(value=5)
        assert counter.get() == 6

    def test_counter_with_labels(self, counter):
        """测试带标签的计数器"""
        labels = {"label1": "value1", "label2": "value2"}
        
        counter.inc(labels=labels)
        assert counter.get(labels=labels) == 1

        counter.inc(labels=labels, value=3)
        assert counter.get(labels=labels) == 4

    def test_counter_multiple_labels(self, counter):
        """测试多个标签组合"""
        labels1 = {"label1": "value1", "label2": "value2"}
        labels2 = {"label1": "value1", "label2": "value3"}
        
        counter.inc(labels=labels1)
        counter.inc(labels=labels2)
        
        assert counter.get(labels=labels1) == 1
        assert counter.get(labels=labels2) == 1

    def test_counter_thread_safety(self, counter):
        """测试计数器线程安全"""
        def increment_worker():
            for _ in range(100):
                counter.inc()
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=increment_worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert counter.get() == 500

class TestGauge:
    """Gauge类测试"""

    @pytest.fixture
    def gauge(self):
        """创建Gauge实例"""
        return Gauge("test_gauge", "Test gauge", ["label1", "label2"])

    def test_gauge_init(self, gauge):
        """测试Gauge初始化"""
        assert gauge.name == "test_gauge"
        assert gauge.description == "Test gauge"
        assert gauge.labels == ["label1", "label2"]

    def test_gauge_set_get(self, gauge):
        """测试仪表盘设置和获取"""
        gauge.set(42.5)
        assert gauge.get() == 42.5

        gauge.set(100.0)
        assert gauge.get() == 100.0

    def test_gauge_with_labels(self, gauge):
        """测试带标签的仪表盘"""
        labels = {"label1": "value1", "label2": "value2"}
        
        gauge.set(25.5, labels=labels)
        assert gauge.get(labels=labels) == 25.5

    def test_gauge_multiple_labels(self, gauge):
        """测试多个标签组合"""
        labels1 = {"label1": "value1", "label2": "value2"}
        labels2 = {"label1": "value1", "label2": "value3"}
        
        gauge.set(10.0, labels=labels1)
        gauge.set(20.0, labels=labels2)
        
        assert gauge.get(labels=labels1) == 10.0
        assert gauge.get(labels=labels2) == 20.0

    def test_gauge_thread_safety(self, gauge):
        """测试仪表盘线程安全"""
        def set_worker():
            for i in range(100):
                gauge.set(i)
        
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=set_worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 最终值应该是最后一个设置的值
        assert gauge.get() == 99

class TestAlertChannel:
    """AlertChannel基类测试"""

    def test_alert_channel_abstract(self):
        """测试AlertChannel是抽象基类"""
        with pytest.raises(TypeError):
            AlertChannel()

class TestDingTalkChannel:
    """DingTalkChannel类测试"""

    @pytest.fixture
    def dingtalk_channel(self):
        """创建钉钉告警通道"""
        return DingTalkChannel("https://oapi.dingtalk.com/robot/send?access_token=test")

    @patch('requests.post')
    def test_dingtalk_send_success(self, mock_post, dingtalk_channel):
        """测试钉钉告警发送成功"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        alert = {
            "name": "Test Alert",
            "level": "WARNING",
            "timestamp": "2024-01-01 12:00:00",
            "data": {"key": "value"}
        }
        
        result = dingtalk_channel.send(alert)
        assert result is True
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_dingtalk_send_failure(self, mock_post, dingtalk_channel):
        """测试钉钉告警发送失败"""
        mock_post.side_effect = Exception("Network error")
        
        alert = {
            "name": "Test Alert",
            "level": "WARNING",
            "timestamp": "2024-01-01 12:00:00",
            "data": {"key": "value"}
        }
        
        result = dingtalk_channel.send(alert)
        assert result is False

class TestWeComChannel:
    """WeComChannel类测试"""

    @pytest.fixture
    def wecom_channel(self):
        """创建企业微信告警通道"""
        return WeComChannel("https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test")

    @patch('requests.post')
    def test_wecom_send_success(self, mock_post, wecom_channel):
        """测试企业微信告警发送成功"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        alert = {
            "name": "Test Alert",
            "level": "WARNING",
            "timestamp": "2024-01-01 12:00:00",
            "data": {"key": "value"}
        }
        
        result = wecom_channel.send(alert)
        assert result is True
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_wecom_send_failure(self, mock_post, wecom_channel):
        """测试企业微信告警发送失败"""
        mock_post.side_effect = Exception("Network error")
        
        alert = {
            "name": "Test Alert",
            "level": "WARNING",
            "timestamp": "2024-01-01 12:00:00",
            "data": {"key": "value"}
        }
        
        result = wecom_channel.send(alert)
        assert result is False

class TestMonitoringSystem:
    """MonitoringSystem类测试"""

    @pytest.fixture
    def monitoring_system(self):
        """创建监控系统实例"""
        return MonitoringSystem(prometheus_port=9091)

    def test_monitoring_system_init(self, monitoring_system):
        """测试监控系统初始化"""
        assert monitoring_system.metrics == {}
        assert monitoring_system.alert_channels == {}
        assert monitoring_system.alert_rules == {}
        assert monitoring_system.prometheus_port == 9091

    def test_register_counter(self, monitoring_system):
        """测试注册计数器"""
        counter = monitoring_system.register_counter(
            "test_counter",
            "Test counter description",
            ["label1", "label2"]
        )
        
        assert isinstance(counter, Counter)
        assert counter.name == "test_counter"
        assert "test_counter" in monitoring_system.metrics

    def test_register_gauge(self, monitoring_system):
        """测试注册仪表盘"""
        gauge = monitoring_system.register_gauge(
            "test_gauge",
            "Test gauge description",
            ["label1", "label2"]
        )
        
        assert isinstance(gauge, Gauge)
        assert gauge.name == "test_gauge"
        assert "test_gauge" in monitoring_system.metrics

    def test_register_metric_duplicate(self, monitoring_system):
        """测试重复注册指标"""
        counter1 = monitoring_system.register_counter("test_counter", "Description")
        counter2 = monitoring_system.register_counter("test_counter", "Description")
        
        assert counter1 is not None
        assert counter2 is None  # 重复注册应该返回None

    def test_add_alert_channel(self, monitoring_system):
        """测试添加告警通道"""
        channel = DingTalkChannel("https://test.com")
        monitoring_system.add_alert_channel("dingtalk", channel)
        
        assert "dingtalk" in monitoring_system.alert_channels
        assert monitoring_system.alert_channels["dingtalk"] is channel

    def test_add_alert_rule(self, monitoring_system):
        """测试添加告警规则"""
        monitoring_system.add_alert_rule(
            "test_rule",
            "counter > 10",
            ["dingtalk"],
            AlertLevel.WARNING
        )
        
        assert "test_rule" in monitoring_system.alert_rules
        rule = monitoring_system.alert_rules["test_rule"]
        assert rule["condition"] == "counter > 10"
        assert rule["channels"] == ["dingtalk"]
        assert rule["level"] == AlertLevel.WARNING

    def test_record_error(self, monitoring_system):
        """测试记录错误"""
        # 先注册错误计数器
        monitoring_system.register_counter("trading_errors_total", "Trading errors")
        
        error = ValueError("Test error")
        monitoring_system.record_error(error, {"context": "test"})
        
        # 检查错误计数器是否增加
        counter = monitoring_system.metrics["trading_errors_total"]
        assert counter.get(labels={"error_type": "ValueError"}) == 1

    @patch('src.infrastructure.monitoring.monitoring_system.MonitoringSystem._should_alert')
    def test_record_error_with_alert(self, mock_should_alert, monitoring_system):
        """测试记录错误并触发告警"""
        mock_should_alert.return_value = True
        
        # 添加告警通道
        mock_channel = Mock()
        monitoring_system.add_alert_channel("test", mock_channel)
        
        error = RuntimeError("Critical error")
        monitoring_system.record_error(error)
        
        # 检查是否触发了告警
        # 这里需要等待告警队列处理
        time.sleep(0.1)

    def test_trigger_alert(self, monitoring_system):
        """测试触发告警"""
        mock_channel = Mock()
        monitoring_system.add_alert_channel("test", mock_channel)
        
        monitoring_system.trigger_alert(
            "TEST_ALERT",
            AlertLevel.WARNING,
            {"key": "value"}
        )
        
        # 检查告警队列
        assert not monitoring_system.alert_queue.empty()

    def test_get_prometheus_metrics(self, monitoring_system):
        """测试获取Prometheus指标"""
        # 注册一些指标
        counter = monitoring_system.register_counter("test_counter", "Test counter")
        gauge = monitoring_system.register_gauge("test_gauge", "Test gauge")
        
        counter.inc()
        gauge.set(42.5)
        
        metrics = monitoring_system.get_prometheus_metrics()
        
        assert "test_counter" in metrics
        assert "test_gauge" in metrics

    def test_monitoring_system_integration(self, monitoring_system):
        """测试监控系统集成功能"""
        # 注册指标
        counter = monitoring_system.register_counter("api_calls_total", "API calls")
        gauge = monitoring_system.register_gauge("response_time", "Response time")
        
        # 模拟API调用
        counter.inc(labels={"endpoint": "/api/v1/data"})
        gauge.set(150.5, labels={"endpoint": "/api/v1/data"})
        
        # 添加告警规则
        monitoring_system.add_alert_rule(
            "high_response_time",
            "response_time > 1000",
            ["dingtalk"],
            AlertLevel.WARNING
        )
        
        # 检查指标值
        assert counter.get(labels={"endpoint": "/api/v1/data"}) == 1
        assert gauge.get(labels={"endpoint": "/api/v1/data"}) == 150.5

    def test_monitoring_system_thread_safety(self, monitoring_system):
        """测试监控系统线程安全"""
        counter = monitoring_system.register_counter("thread_test", "Thread test")
        
        def worker():
            for _ in range(100):
                counter.inc()
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert counter.get() == 500

def test_monitoring_system_performance():
    """测试监控系统性能"""
    monitoring_system = MonitoringSystem()
    
    # 注册大量指标
    counters = []
    gauges = []
    
    for i in range(100):
        counter = monitoring_system.register_counter(f"counter_{i}", f"Counter {i}")
        gauge = monitoring_system.register_gauge(f"gauge_{i}", f"Gauge {i}")
        counters.append(counter)
        gauges.append(gauge)
    
    # 模拟大量操作
    start_time = time.time()
    
    for i in range(1000):
        counter = counters[i % 100]
        gauge = gauges[i % 100]
        counter.inc()
        gauge.set(i)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 验证性能
    assert execution_time < 1.0  # 应该在1秒内完成
    
    # 验证数据
    for i in range(100):
        assert counters[i].get() == 10  # 每个计数器应该被增加10次
        assert gauges[i].get() == 990  # 最后一个设置的值 