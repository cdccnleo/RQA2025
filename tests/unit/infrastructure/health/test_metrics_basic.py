"""
Metrics模块基础测试套件

针对metrics.py模块的基础测试覆盖
目标: 建立基础测试框架，从0%覆盖率开始
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock
from datetime import datetime

# 导入被测试模块
from src.infrastructure.health.models.metrics import (
    MetricsCollector,
    MetricType
)


class TestMetricsBasic:
    """Metrics基础测试"""

    @pytest.fixture
    def metrics_collector(self):
        """创建MetricsCollector实例"""
        return MetricsCollector("test_collector")

    def test_metric_type_enum(self):
        """测试MetricType枚举"""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"

    def test_metrics_collector_initialization(self, metrics_collector):
        """测试MetricsCollector初始化"""
        assert metrics_collector.name == "test_collector"
        assert metrics_collector._initialized == False
        assert isinstance(metrics_collector._metrics_data, dict)
        assert metrics_collector._collection_count == 0
        assert metrics_collector._last_collection_time is None

    def test_metrics_collector_default_name(self):
        """测试MetricsCollector默认名称"""
        collector = MetricsCollector()
        assert collector.name == "MetricsCollector"

    def test_record_metric_basic(self, metrics_collector):
        """测试基本指标记录"""
        # 记录指标不应该抛出异常
        metrics_collector.record_metric("test_metric", 42.0)

        # 验证计数器没有变化（因为是抽象方法）
        assert metrics_collector._collection_count == 0

    def test_test_connection_basic(self, metrics_collector):
        """测试基本连接测试"""
        result = metrics_collector.test_connection()

        # 默认实现应该返回True
        assert result == True

    def test_get_status(self, metrics_collector):
        """测试获取状态信息"""
        status = metrics_collector.get_status()

        assert isinstance(status, dict)
        assert 'name' in status
        assert 'initialized' in status
        assert 'collection_count' in status
        assert status['name'] == "test_collector"
        assert status['initialized'] == False
        assert status['collection_count'] == 0

    def test_get_component_info(self, metrics_collector):
        """测试获取组件信息"""
        info = metrics_collector.get_component_info()

        assert isinstance(info, dict)
        assert 'component_type' in info
        assert 'name' in info
        assert info['component_type'] == "MetricsCollector"

    def test_is_healthy_not_initialized(self, metrics_collector):
        """测试未初始化时的健康状态"""
        # 根据实际实现，未初始化时应该返回False
        assert metrics_collector.is_healthy() == False

    def test_initialize(self, metrics_collector):
        """测试初始化"""
        result = metrics_collector.initialize()

        assert result == True
        assert metrics_collector._initialized == True
        assert metrics_collector._collection_count == 0
        assert metrics_collector._last_collection_time is None

    def test_shutdown(self, metrics_collector):
        """测试关闭"""
        # 先初始化
        metrics_collector.initialize()

        result = metrics_collector.shutdown()

        assert result == True
        assert metrics_collector._initialized == False

    def test_is_healthy_after_initialize(self, metrics_collector):
        """测试初始化后的健康状态"""
        metrics_collector.initialize()

        assert metrics_collector.is_healthy() == True

    def test_metric_definition(self):
        """测试MetricDefinition类"""
        from src.infrastructure.health.models.metrics import MetricDefinition

        definition = MetricDefinition("test_metric", MetricType.COUNTER, "Test metric", ["label1", "label2"])

        assert definition.name == "test_metric"
        assert definition.metric_type == MetricType.COUNTER
        assert definition.description == "Test metric"
        assert definition.labels == ["label1", "label2"]

    def test_metric_definition_to_dict(self):
        """测试MetricDefinition的to_dict方法"""
        from src.infrastructure.health.models.metrics import MetricDefinition

        definition = MetricDefinition("test_metric", MetricType.GAUGE, "Test metric")
        result = definition.to_dict()

        assert isinstance(result, dict)
        assert result['name'] == "test_metric"
        assert result['type'] == "gauge"
        assert result['description'] == "Test metric"
        assert result['labels'] == []

    def test_predefined_metrics(self):
        """测试预定义指标"""
        from src.infrastructure.health.models.metrics import PERFORMANCE_METRICS, BUSINESS_METRICS, SYSTEM_METRICS

        assert len(PERFORMANCE_METRICS) == 3
        assert len(BUSINESS_METRICS) == 3
        assert len(SYSTEM_METRICS) == 3

        # 检查第一个性能指标
        assert PERFORMANCE_METRICS[0].name == "execution_time"
        assert PERFORMANCE_METRICS[0].metric_type == MetricType.HISTOGRAM
