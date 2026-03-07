#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataQualityMonitor 全面单元测试
覆盖计算指标、阈值检查、历史记录、汇总报告、告警等核心路径。
"""

import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.quality.data_quality_monitor import (
    AccuracyRule,
    AlertLevel,
    AnomalyRecord,
    CompletenessRule,
    ConsistencyRule,
    DataQualityMonitor,
    DataQualityRule,
    QualityCheckResult,
    QualityLevel,
    QualityMetric,
    QualityReport,
    TimelinessRule,
)


@pytest.fixture
def monitor():
    """DataQualityMonitor 实例"""
    return DataQualityMonitor(data_source="test_source")


@pytest.fixture
def sample_dataframe():
    """示例数据框"""
    return pd.DataFrame({
        "price": [100.0, 101.0, 102.0, 103.0, 104.0],
        "volume": [1000, 1100, 1200, 1300, 1400],
        "timestamp": pd.date_range(
            pd.Timestamp.now() - timedelta(hours=4), periods=5, freq="h"
        ),
    })


@pytest.fixture
def poor_dataframe():
    """包含缺失值的数据框"""
    return pd.DataFrame({
        "price": [100.0, np.nan, np.nan, 103.0, 104.0],
        "volume": [1000, np.nan, 1200, np.nan, 1400],
        "timestamp": pd.date_range(
            pd.Timestamp.now() - timedelta(hours=4), periods=5, freq="h"
        ),
    })


class TestCalculateMetrics:
    """测试计算指标方法"""

    def test_calculate_metrics_normal(self, monitor, sample_dataframe):
        """测试正常数据的指标计算"""
        metrics = monitor.calculate_metrics(sample_dataframe)

        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

    def test_calculate_metrics_empty_data(self, monitor):
        """测试空数据的指标计算"""
        empty_df = pd.DataFrame()
        metrics = monitor.calculate_metrics(empty_df)

        assert isinstance(metrics, dict)
        # 空数据应该导致某些指标为0
        assert any(v == 0.0 for v in metrics.values())

    def test_calculate_metrics_rule_exception(self, monitor, sample_dataframe):
        """测试规则执行异常时的处理"""
        # 创建一个会抛出异常的规则
        class ExceptionRule(DataQualityRule):
            def __init__(self):
                super().__init__("ExceptionRule")
            
            def check(self, data):
                raise ValueError("Rule failed")

        monitor.add_rule(ExceptionRule())
        metrics = monitor.calculate_metrics(sample_dataframe)

        assert "ExceptionRule" in metrics
        assert metrics["ExceptionRule"] == 0.0


class TestCheckThresholds:
    """测试阈值检查方法"""

    def test_check_thresholds_normal(self, monitor, sample_dataframe):
        """测试正常数据的阈值检查"""
        violations = monitor.check_thresholds(sample_dataframe)

        assert isinstance(violations, list)
        # 高质量数据可能没有违规
        assert all(isinstance(v, dict) for v in violations)

    def test_check_thresholds_with_violations(self, monitor, poor_dataframe):
        """测试存在违规的阈值检查"""
        violations = monitor.check_thresholds(poor_dataframe)

        assert isinstance(violations, list)
        # 低质量数据可能有违规


class TestRecordMetrics:
    """测试记录指标方法"""

    def test_record_metrics_basic(self, monitor, sample_dataframe):
        """测试基本指标记录"""
        initial_count = len(monitor.metrics_history)

        entry = monitor.record_metrics(sample_dataframe)

        assert len(monitor.metrics_history) == initial_count + 1
        assert "timestamp" in entry
        assert "metrics" in entry
        assert "data_source" in entry
        assert entry["data_source"] == "test_source"

    def test_record_metrics_with_custom_timestamp(self, monitor, sample_dataframe):
        """测试使用自定义时间戳记录指标"""
        custom_time = datetime.now() - timedelta(days=1)

        entry = monitor.record_metrics(sample_dataframe, timestamp=custom_time)

        assert entry["timestamp"] == custom_time

    def test_record_metrics_with_custom_source(self, monitor, sample_dataframe):
        """测试使用自定义数据源记录指标"""
        entry = monitor.record_metrics(sample_dataframe, data_source="custom_source")

        assert entry["data_source"] == "custom_source"


class TestGetMetricsHistory:
    """测试获取指标历史方法"""

    def test_get_metrics_history_default(self, monitor, sample_dataframe):
        """测试获取所有历史指标"""
        # 记录多个指标
        for i in range(3):
            monitor.record_metrics(sample_dataframe)

        history = monitor.get_metrics_history()

        assert len(history) >= 3
        assert all("timestamp" in h for h in history)
        assert all("metrics" in h for h in history)

    def test_get_metrics_history_with_limit(self, monitor, sample_dataframe):
        """测试使用限制获取历史指标"""
        # 记录5个指标
        for i in range(5):
            monitor.record_metrics(sample_dataframe)

        history = monitor.get_metrics_history(limit=3)

        assert len(history) == 3
        # 应该返回最近的3个

    def test_get_metrics_history_empty(self, monitor):
        """测试空历史"""
        history = monitor.get_metrics_history()

        assert isinstance(history, list)
        assert len(history) == 0


class TestRegisterAlertHandler:
    """测试注册告警处理器"""

    def test_register_alert_handler(self, monitor):
        """测试注册告警处理器"""
        handler = Mock()

        monitor.register_alert_handler(handler)

        assert handler in monitor.alert_handlers

    def test_register_alert_handler_duplicate(self, monitor):
        """测试重复注册处理器（不应重复添加）"""
        handler = Mock()

        monitor.register_alert_handler(handler)
        monitor.register_alert_handler(handler)

        assert monitor.alert_handlers.count(handler) == 1


class TestGetQualityHistory:
    """测试获取质量历史方法"""

    def test_get_quality_history_with_reports(self, monitor, sample_dataframe):
        """测试获取有报告的质量历史"""
        # 生成多个报告
        for i in range(3):
            monitor.check_quality(sample_dataframe, f"source_{i}")

        history = monitor.get_quality_history(days=7)

        assert len(history) >= 3
        assert all(isinstance(h, QualityCheckResult) for h in history)

    def test_get_quality_history_filtered_by_days(self, monitor, sample_dataframe):
        """测试按天数过滤历史"""
        # 生成一个报告
        monitor.check_quality(sample_dataframe, "recent")

        # 获取最近1天的历史
        history = monitor.get_quality_history(days=1)

        assert len(history) >= 1


class TestGetAnomalyHistory:
    """测试获取异常历史方法"""

    def test_get_anomaly_history_with_anomalies(self, monitor, poor_dataframe):
        """测试获取有异常的历史"""
        # 生成包含异常的报告
        monitor.check_quality(poor_dataframe, "poor_source")

        history = monitor.get_anomaly_history(days=7)

        assert isinstance(history, list)
        assert all(isinstance(a, AnomalyRecord) for a in history)

    def test_get_anomaly_history_filtered(self, monitor, poor_dataframe):
        """测试过滤的异常历史"""
        monitor.check_quality(poor_dataframe, "source")

        history = monitor.get_anomaly_history(days=1)

        assert isinstance(history, list)


class TestGenerateSummaryReport:
    """测试生成汇总报告方法"""

    def test_generate_summary_report_with_history(self, monitor, sample_dataframe):
        """测试有历史时的汇总报告生成"""
        # 生成多个报告
        for i in range(3):
            monitor.check_quality(sample_dataframe, f"source_{i}")

        summary = monitor.generate_summary_report(days=7)

        assert "period_days" in summary
        assert "total_reports" in summary
        assert "average_score" in summary
        assert "min_score" in summary
        assert "max_score" in summary
        assert summary["total_reports"] >= 3

    def test_generate_summary_report_empty(self, monitor):
        """测试无历史时的汇总报告"""
        summary = monitor.generate_summary_report(days=7)

        assert "message" in summary
        assert summary["message"] == "没有历史数据"


class TestExportReport:
    """测试导出报告方法"""

    def test_export_report_json(self, monitor, sample_dataframe):
        """测试导出JSON格式报告"""
        report_obj = monitor.check_quality(sample_dataframe, "test")
        report = report_obj.quality_report

        json_str = monitor.export_report(report, format="json")

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert "timestamp" in data or "data_source" in data

    def test_export_report_csv(self, monitor, sample_dataframe):
        """测试导出CSV格式报告"""
        report_obj = monitor.check_quality(sample_dataframe, "test")
        report = report_obj.quality_report

        csv_str = monitor.export_report(report, format="csv")

        assert isinstance(csv_str, str)
        assert "," in csv_str  # CSV格式应该包含逗号

    def test_export_report_invalid_format(self, monitor, sample_dataframe):
        """测试无效格式的导出"""
        report_obj = monitor.check_quality(sample_dataframe, "test")
        report = report_obj.quality_report

        with pytest.raises(ValueError, match="不支持的导出格式"):
            monitor.export_report(report, format="invalid")


class TestTrackMetrics:
    """测试跟踪指标方法（向后兼容）"""

    def test_track_metrics_with_dataframe(self, monitor, sample_dataframe):
        """测试使用DataFrame跟踪指标"""
        result = monitor.track_metrics(sample_dataframe, data_source="df_test")

        assert isinstance(result, dict)
        # track_metrics 返回格式可能不同，检查基本字段
        assert "data_source" in result
        assert "timestamp" in result
        assert "quality_level" in result or "score" in result or "overall_score" in result

    def test_track_metrics_with_model(self, monitor):
        """测试使用数据模型跟踪指标"""
        # 创建一个模拟数据模型
        class MockDataModel:
            def __init__(self):
                self.data = pd.DataFrame({"value": [1, 2, 3]})

        model = MockDataModel()
        result = monitor.track_metrics(model, data_source="model_test")

        assert isinstance(result, dict)

    def test_track_metrics_with_invalid_data(self, monitor):
        """测试使用无效数据跟踪指标"""
        result = monitor.track_metrics(None, data_source="invalid")

        assert isinstance(result, dict)
        # track_metrics 对无效数据可能返回不同的格式
        assert result.get("quality_level") == "critical" or result.get("score") == 0.0 or result.get("overall_score") == 0.0


class TestQualityCheckResult:
    """测试QualityCheckResult类"""

    def test_quality_check_result_attributes(self, monitor, sample_dataframe):
        """测试QualityCheckResult的属性访问"""
        result = monitor.check_quality(sample_dataframe, "test")

        assert isinstance(result, QualityCheckResult)
        assert hasattr(result, "overall_score")
        assert hasattr(result, "quality_level")
        assert hasattr(result, "quality_report")

    def test_quality_check_result_to_dict(self, monitor, sample_dataframe):
        """测试转换为字典"""
        result = monitor.check_quality(sample_dataframe, "test")

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "overall_score" in result_dict


class TestRecommendationsAndAlerts:
    """测试建议和告警生成"""

    def test_generate_recommendations_poor_quality(self, monitor, poor_dataframe):
        """测试低质量数据的建议生成"""
        report_obj = monitor.check_quality(poor_dataframe, "poor")

        assert len(report_obj.recommendations) > 0
        assert all(isinstance(rec, str) for rec in report_obj.recommendations)

    @patch("src.data.quality.data_quality_monitor.logger")
    def test_trigger_alert_critical(self, mock_logger, monitor):
        """测试触发严重告警"""
        monitor.alert_config["enabled"] = True
        monitor.alert_config["levels"] = {
            "critical": {"threshold": 0.6, "channels": ["log"]}
        }

        # 创建一个低质量报告以触发告警
        poor_data = pd.DataFrame({"value": [np.nan] * 10})
        report_obj = monitor.check_quality(poor_data, "critical_source")

        # 验证日志被调用（如果是log channel）
        if report_obj.alert_level == AlertLevel.CRITICAL:
            mock_logger.warning.assert_called()


class TestMonitorInitializationVariants:
    """测试监控器初始化的不同变体"""

    def test_init_with_legacy_dict(self):
        """测试使用旧式字典参数初始化"""
        config = {"data_source": "legacy", "alert_enabled": False}
        monitor = DataQualityMonitor(config)

        assert monitor.data_source == "legacy"
        assert monitor.alert_enabled is False

    def test_init_with_metrics_enabled(self):
        """测试指定启用的指标"""
        config = {"metrics_enabled": ["completeness", "accuracy"]}
        monitor = DataQualityMonitor(config=config)

        assert "completeness" in monitor.metrics_enabled
        assert "accuracy" in monitor.metrics_enabled
        assert len(monitor.rules) == 2

    def test_init_with_empty_metrics_enabled(self):
        """测试空指标列表时使用默认值"""
        config = {"metrics_enabled": []}
        monitor = DataQualityMonitor(config=config)

        assert len(monitor.metrics_enabled) == 4  # 应该使用默认值
        assert len(monitor.rules) == 4


class TestRulesWithSpecificColumns:
    """测试指定列的规则"""

    def test_completeness_rule_with_specific_columns(self, monitor, poor_dataframe):
        """测试指定列的完整性规则"""
        rule = CompletenessRule(columns=["price"], threshold=0.8)
        monitor.add_rule(rule)

        result = monitor.check_quality(poor_dataframe, "test")

        assert "completeness" in result.metrics

    def test_accuracy_rule_with_numeric_columns(self, monitor, sample_dataframe):
        """测试指定数值列的准确性规则"""
        rule = AccuracyRule(numeric_columns=["price", "volume"], threshold=0.8)
        monitor.add_rule(rule)

        result = monitor.check_quality(sample_dataframe, "test")

        assert "accuracy" in result.metrics


class TestAutoRepairFeature:
    """测试自动修复功能"""

    def test_auto_repair_enabled(self):
        """测试启用自动修复"""
        config = {"auto_repair": True}
        monitor = DataQualityMonitor(config=config)

        assert monitor.auto_repair is True

    def test_auto_repair_disabled(self):
        """测试禁用自动修复"""
        config = {"auto_repair": False}
        monitor = DataQualityMonitor(config=config)

        assert monitor.auto_repair is False

