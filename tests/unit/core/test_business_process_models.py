#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试业务流程数据模型

测试目标：提升business_process/models/models.py的覆盖率到100%
"""

import pytest

# 尝试导入所需模块
try:
    from src.core.business_process.models.models import (
        ProcessConfig,
        ProcessInstance,
        ProcessMetrics,
        PerformanceMetrics
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
from datetime import datetime
from dataclasses import fields

from src.core.business_process.models.models import (
    ProcessConfig,
    ProcessInstance,
    ProcessMetrics,
    PerformanceMetrics
)
from src.core.business_process.config.enums import BusinessProcessState


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestProcessConfig:
    """测试流程配置"""

    @pytest.fixture
    def process_config(self):
        """创建流程配置实例"""
        return ProcessConfig(
            process_id="test_process_001",
            process_name="Test Process",
            description="A test process configuration"
        )

    def test_process_config_creation(self, process_config):
        """测试流程配置创建"""
        assert process_config.process_id == "test_process_001"
        assert process_config.process_name == "Test Process"
        assert process_config.description == "A test process configuration"

    def test_process_config_default_values(self):
        """测试流程配置默认值"""
        config = ProcessConfig(
            process_id="minimal_config",
            process_name="Minimal Config"
        )

        assert config.process_id == "minimal_config"
        assert config.process_name == "Minimal Config"
        assert config.description is None
        assert config.data_sources == []
        assert config.data_quality_rules == {}
        assert config.feature_config == {}
        assert config.gpu_acceleration_enabled == False
        assert config.model_configs == []
        assert config.ensemble_method is None
        assert config.strategy_configs == []
        assert config.risk_check_configs == {}
        assert config.trading_configs == {}
        assert config.monitoring_configs == {}
        assert config.timeout_seconds == 300
        assert config.retry_count == 3

    def test_process_config_with_custom_values(self):
        """测试流程配置自定义值"""
        config = ProcessConfig(
            process_id="custom_config",
            process_name="Custom Config",
            description="Custom configuration",
            data_sources=[{"type": "database", "connection": "sqlite:///test.db"}],
            gpu_acceleration_enabled=True,
            timeout_seconds=600,
            retry_count=5
        )

        assert config.data_sources == [{"type": "database", "connection": "sqlite:///test.db"}]
        assert config.gpu_acceleration_enabled == True
        assert config.timeout_seconds == 600
        assert config.retry_count == 5

    def test_process_config_metadata(self, process_config):
        """测试流程配置元数据"""
        # 检查是否包含元数据字段
        field_names = [f.name for f in fields(process_config)]
        assert "metadata" in field_names

        # 元数据应该是字典类型
        assert isinstance(process_config.metadata, dict)

    def test_process_config_equality(self):
        """测试流程配置相等性"""
        config1 = ProcessConfig(
            process_id="test",
            process_name="Test",
            timeout_seconds=300
        )
        config2 = ProcessConfig(
            process_id="test",
            process_name="Test",
            timeout_seconds=300
        )
        config3 = ProcessConfig(
            process_id="different",
            process_name="Test",
            timeout_seconds=300
        )

        assert config1 == config2
        assert config1 != config3

    def test_process_config_string_representation(self, process_config):
        """测试流程配置字符串表示"""
        str_repr = str(process_config)
        assert "test_process_001" in str_repr
        assert "Test Process" in str_repr


class TestProcessInstance:
    """测试流程实例"""

    @pytest.fixture
    def process_instance(self):
        """创建流程实例"""
        return ProcessInstance(
            instance_id="instance_001",
            process_id="process_001",
            state=BusinessProcessState.INITIALIZED
        )

    def test_process_instance_creation(self, process_instance):
        """测试流程实例创建"""
        assert process_instance.instance_id == "instance_001"
        assert process_instance.process_id == "process_001"
        assert process_instance.state == BusinessProcessState.INITIALIZED

    def test_process_instance_default_values(self):
        """测试流程实例默认值"""
        instance = ProcessInstance(
            instance_id="minimal_instance",
            process_id="minimal_process"
        )

        assert instance.instance_id == "minimal_instance"
        assert instance.process_id == "minimal_process"
        assert instance.state == BusinessProcessState.CREATED
        assert instance.start_time is None
        assert instance.end_time is None
        assert instance.current_step == ""
        assert instance.progress == 0.0
        assert instance.error_message == ""
        assert instance.retry_count == 0

    def test_process_instance_with_execution_data(self):
        """测试流程实例执行数据"""
        start_time = datetime.now()

        instance = ProcessInstance(
            instance_id="executing_instance",
            process_id="test_process",
            state=BusinessProcessState.RUNNING,
            start_time=start_time,
            current_step="data_processing",
            progress=0.5,
            retry_count=1
        )

        assert instance.state == BusinessProcessState.RUNNING
        assert instance.start_time == start_time
        assert instance.current_step == "data_processing"
        assert instance.progress == 0.5
        assert instance.retry_count == 1

    def test_process_instance_with_error(self):
        """测试流程实例错误状态"""
        instance = ProcessInstance(
            instance_id="error_instance",
            process_id="test_process",
            state=BusinessProcessState.ERROR,
            error_message="Process failed due to data validation error"
        )

        assert instance.state == BusinessProcessState.ERROR
        assert "validation error" in instance.error_message

    def test_process_instance_metadata(self, process_instance):
        """测试流程实例元数据"""
        field_names = [f.name for f in fields(process_instance)]
        assert "metadata" in field_names
        assert isinstance(process_instance.metadata, dict)

    def test_process_instance_results(self, process_instance):
        """测试流程实例结果"""
        field_names = [f.name for f in fields(process_instance)]
        assert "results" in field_names
        assert isinstance(process_instance.results, dict)

    def test_process_instance_step_history(self, process_instance):
        """测试流程实例步骤历史"""
        field_names = [f.name for f in fields(process_instance)]
        assert "step_history" in field_names
        assert isinstance(process_instance.step_history, list)

    def test_process_instance_equality(self):
        """测试流程实例相等性"""
        instance1 = ProcessInstance(
            instance_id="test",
            process_id="process",
            state=BusinessProcessState.RUNNING
        )
        instance2 = ProcessInstance(
            instance_id="test",
            process_id="process",
            state=BusinessProcessState.RUNNING
        )
        instance3 = ProcessInstance(
            instance_id="different",
            process_id="process",
            state=BusinessProcessState.RUNNING
        )

        assert instance1 == instance2
        assert instance1 != instance3


class TestProcessMetrics:
    """测试流程指标"""

    @pytest.fixture
    def process_metrics(self):
        """创建流程指标实例"""
        return ProcessMetrics(
            process_id="test_process",
            total_executions=100,
            successful_executions=95,
            failed_executions=5,
            average_execution_time=45.5
        )

    def test_process_metrics_creation(self, process_metrics):
        """测试流程指标创建"""
        assert process_metrics.process_id == "test_process"
        assert process_metrics.total_executions == 100
        assert process_metrics.successful_executions == 95
        assert process_metrics.failed_executions == 5
        assert process_metrics.average_execution_time == 45.5

    def test_process_metrics_calculated_fields(self, process_metrics):
        """测试流程指标计算字段"""
        # 成功率应该是95/100 = 0.95
        assert process_metrics.success_rate == 0.95

        # 失败率应该是5/100 = 0.05
        assert process_metrics.failure_rate == 0.05

    def test_process_metrics_with_zero_executions(self):
        """测试零执行的流程指标"""
        metrics = ProcessMetrics(
            process_id="empty_process",
            total_executions=0,
            successful_executions=0,
            failed_executions=0,
            average_execution_time=0.0
        )

        # 避免除零错误
        assert metrics.success_rate == 0.0
        assert metrics.failure_rate == 0.0

    def test_process_metrics_additional_fields(self):
        """测试流程指标额外字段"""
        metrics = ProcessMetrics(
            process_id="detailed_process",
            total_executions=50,
            successful_executions=48,
            failed_executions=2,
            average_execution_time=30.0,
            max_execution_time=120.0,
            min_execution_time=15.0,
            median_execution_time=28.0
        )

        assert metrics.max_execution_time == 120.0
        assert metrics.min_execution_time == 15.0
        assert metrics.median_execution_time == 28.0
        assert metrics.success_rate == 0.96  # 48/50

    def test_process_metrics_default_values(self):
        """测试流程指标默认值"""
        metrics = ProcessMetrics(
            process_id="minimal_metrics",
            total_executions=10,
            successful_executions=8,
            failed_executions=2,
            average_execution_time=25.0
        )

        assert metrics.max_execution_time == 0.0
        assert metrics.min_execution_time == 0.0
        assert metrics.median_execution_time == 0.0
        assert metrics.last_execution_time is None

    def test_process_metrics_string_representation(self, process_metrics):
        """测试流程指标字符串表示"""
        str_repr = str(process_metrics)
        assert "test_process" in str_repr
        assert "100" in str_repr
        assert "95" in str_repr


class TestPerformanceMetrics:
    """测试性能指标"""

    @pytest.fixture
    def performance_metrics(self):
        """创建性能指标实例"""
        return PerformanceMetrics(
            metric_name="cpu_usage",
            value=75.5,
            unit="percent",
            timestamp=datetime.now()
        )

    def test_performance_metrics_creation(self, performance_metrics):
        """测试性能指标创建"""
        assert performance_metrics.metric_name == "cpu_usage"
        assert performance_metrics.value == 75.5
        assert performance_metrics.unit == "percent"
        assert isinstance(performance_metrics.timestamp, datetime)

    def test_performance_metrics_with_category(self):
        """测试带类别的性能指标"""
        metrics = PerformanceMetrics(
            metric_name="memory_usage",
            value=1024.0,
            unit="MB",
            timestamp=datetime.now(),
            category="system"
        )

        assert metrics.category == "system"

    def test_performance_metrics_default_category(self):
        """测试性能指标默认类别"""
        metrics = PerformanceMetrics(
            metric_name="response_time",
            value=150.0,
            unit="ms",
            timestamp=datetime.now()
        )

        assert metrics.category == "performance"

    def test_performance_metrics_additional_fields(self):
        """测试性能指标额外字段"""
        metrics = PerformanceMetrics(
            metric_name="throughput",
            value=1000.0,
            unit="requests_per_second",
            timestamp=datetime.now(),
            category="throughput",
            tags={"endpoint": "/api/trade", "method": "POST"}
        )

        assert metrics.tags == {"endpoint": "/api/trade", "method": "POST"}

    def test_performance_metrics_default_tags(self, performance_metrics):
        """测试性能指标默认标签"""
        assert performance_metrics.tags == {}

    def test_performance_metrics_equality(self):
        """测试性能指标相等性"""
        timestamp = datetime.now()

        metrics1 = PerformanceMetrics("test", 100.0, "units", timestamp)
        metrics2 = PerformanceMetrics("test", 100.0, "units", timestamp)
        metrics3 = PerformanceMetrics("different", 100.0, "units", timestamp)

        assert metrics1 == metrics2
        assert metrics1 != metrics3

    def test_performance_metrics_string_representation(self, performance_metrics):
        """测试性能指标字符串表示"""
        str_repr = str(performance_metrics)
        assert "cpu_usage" in str_repr
        assert "75.5" in str_repr
        assert "percent" in str_repr


class TestBusinessProcessModelsIntegration:
    """测试业务流程模型集成"""

    def test_complete_process_lifecycle(self):
        """测试完整流程生命周期"""
        # 1. 创建流程配置
        config = ProcessConfig(
            process_id="trading_process",
            process_name="Automated Trading Process",
            description="Complete trading workflow",
            data_sources=[
                {"type": "market_data", "source": "real_time"},
                {"type": "historical_data", "period": "1year"}
            ],
            model_configs=[
                {"type": "xgboost", "params": {"max_depth": 6}},
                {"type": "neural_network", "layers": [64, 32]}
            ],
            strategy_configs=[
                {"name": "momentum", "threshold": 0.02},
                {"name": "mean_reversion", "window": 20}
            ],
            timeout_seconds=600
        )

        assert config.process_id == "trading_process"
        assert len(config.data_sources) == 2
        assert len(config.model_configs) == 2
        assert len(config.strategy_configs) == 2

        # 2. 创建流程实例
        instance = ProcessInstance(
            instance_id="trading_run_001",
            process_id=config.process_id,
            state=BusinessProcessState.CREATED
        )

        assert instance.process_id == config.process_id
        assert instance.state == BusinessProcessState.CREATED

        # 3. 模拟执行过程
        instance.state = BusinessProcessState.RUNNING
        instance.start_time = datetime.now()
        instance.current_step = "data_loading"
        instance.progress = 0.2

        assert instance.state == BusinessProcessState.RUNNING
        assert instance.current_step == "data_loading"
        assert instance.progress == 0.2

        # 4. 完成执行
        instance.current_step = "strategy_execution"
        instance.progress = 0.8

        instance.state = BusinessProcessState.COMPLETED
        instance.end_time = datetime.now()
        instance.results = {
            "total_trades": 50,
            "pnl": 2500.0,
            "win_rate": 0.68,
            "sharpe_ratio": 1.8
        }

        assert instance.state == BusinessProcessState.COMPLETED
        assert instance.progress == 0.8
        assert instance.results["total_trades"] == 50

    def test_process_metrics_aggregation(self):
        """测试流程指标聚合"""
        # 创建多个流程指标
        metrics_list = [
            ProcessMetrics("process_A", 100, 90, 10, 45.0),
            ProcessMetrics("process_B", 80, 75, 5, 38.0),
            ProcessMetrics("process_C", 120, 105, 15, 52.0)
        ]

        # 计算总体指标
        total_executions = sum(m.total_executions for m in metrics_list)
        total_successful = sum(m.successful_executions for m in metrics_list)
        total_failed = sum(m.failed_executions for m in metrics_list)

        overall_success_rate = total_successful / total_executions if total_executions > 0 else 0

        assert total_executions == 300
        assert total_successful == 270
        assert total_failed == 30
        assert overall_success_rate == 0.9  # 270/300

        # 验证每个指标的成功率
        for metrics in metrics_list:
            expected_rate = metrics.successful_executions / metrics.total_executions
            assert metrics.success_rate == expected_rate

    def test_performance_metrics_collection(self):
        """测试性能指标收集"""
        # 模拟收集各种性能指标
        metrics_list = [
            PerformanceMetrics("cpu_usage", 65.5, "percent", datetime.now(), "system"),
            PerformanceMetrics("memory_usage", 2048.0, "MB", datetime.now(), "system"),
            PerformanceMetrics("disk_io", 150.0, "MB/s", datetime.now(), "storage"),
            PerformanceMetrics("network_io", 500.0, "Mbps", datetime.now(), "network"),
            PerformanceMetrics("response_time", 45.0, "ms", datetime.now(), "performance"),
            PerformanceMetrics("throughput", 1200.0, "req/s", datetime.now(), "performance")
        ]

        # 按类别分组
        system_metrics = [m for m in metrics_list if m.category == "system"]
        performance_metrics = [m for m in metrics_list if m.category == "performance"]

        assert len(system_metrics) == 2
        assert len(performance_metrics) == 2

        # 验证系统指标
        cpu_metric = next(m for m in system_metrics if m.metric_name == "cpu_usage")
        memory_metric = next(m for m in system_metrics if m.metric_name == "memory_usage")

        assert cpu_metric.value == 65.5
        assert cpu_metric.unit == "percent"
        assert memory_metric.value == 2048.0
        assert memory_metric.unit == "MB"

        # 验证性能指标
        response_metric = next(m for m in performance_metrics if m.metric_name == "response_time")
        throughput_metric = next(m for m in performance_metrics if m.metric_name == "throughput")

        assert response_metric.value == 45.0
        assert response_metric.unit == "ms"
        assert throughput_metric.value == 1200.0
        assert throughput_metric.unit == "req/s"

    def test_process_error_handling(self):
        """测试流程错误处理"""
        # 创建一个失败的流程实例
        failed_instance = ProcessInstance(
            instance_id="failed_instance",
            process_id="test_process",
            state=BusinessProcessState.ERROR,
            error_message="Data validation failed: missing required fields",
            retry_count=2
        )

        assert failed_instance.state == BusinessProcessState.ERROR
        assert "validation failed" in failed_instance.error_message
        assert failed_instance.retry_count == 2

        # 记录错误到步骤历史
        failed_instance.step_history.append({
            "step": "data_validation",
            "status": "failed",
            "error": failed_instance.error_message,
            "timestamp": datetime.now()
        })

        assert len(failed_instance.step_history) == 1
        assert failed_instance.step_history[0]["status"] == "failed"

    def test_process_monitoring_and_metrics(self):
        """测试流程监控和指标"""
        # 创建流程指标
        metrics = ProcessMetrics(
            process_id="monitored_process",
            total_executions=1000,
            successful_executions=950,
            failed_executions=50,
            average_execution_time=42.5,
            max_execution_time=180.0,
            min_execution_time=12.0,
            median_execution_time=40.0
        )

        # 验证计算指标
        assert metrics.success_rate == 0.95  # 950/1000
        assert metrics.failure_rate == 0.05  # 50/1000

        # 验证性能指标
        assert metrics.average_execution_time == 42.5
        assert metrics.max_execution_time == 180.0
        assert metrics.min_execution_time == 12.0
        assert metrics.median_execution_time == 40.0

        # 创建对应的性能指标
        perf_metrics = [
            PerformanceMetrics("execution_time", metrics.average_execution_time, "seconds",
                             datetime.now(), "performance", {"process_id": metrics.process_id}),
            PerformanceMetrics("success_rate", metrics.success_rate, "ratio",
                             datetime.now(), "quality", {"process_id": metrics.process_id})
        ]

        # 验证性能指标
        exec_time_metric = perf_metrics[0]
        success_rate_metric = perf_metrics[1]

        assert exec_time_metric.metric_name == "execution_time"
        assert exec_time_metric.value == 42.5
        assert exec_time_metric.tags["process_id"] == "monitored_process"

        assert success_rate_metric.metric_name == "success_rate"
        assert success_rate_metric.value == 0.95
        assert success_rate_metric.category == "quality"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
