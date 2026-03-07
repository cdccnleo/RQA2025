# -*- coding: utf-8 -*-
"""
数据质量监控器测试
测试数据质量检查、异常检测和报告功能
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


import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from src.data.quality.data_quality_monitor import (
    DataQualityMonitor,
    QualityLevel,
    AlertLevel,
    QualityMetric,
    QualityReport,
    AnomalyRecord,
    DataQualityRule,
    CompletenessRule,
    ConsistencyRule,
    AccuracyRule,
    TimelinessRule
)


class TestDataQualityMonitorInitialization:
    """测试数据质量监控器初始化"""

    def test_init_with_default_config(self):
        """测试使用默认配置初始化"""
        monitor = DataQualityMonitor()
        assert monitor is not None
        assert monitor.config == {}
        assert len(monitor.rules) == 4  # 默认4个规则
        assert monitor.alert_config['enabled'] is True

    def test_init_with_custom_config(self):
        """测试使用自定义配置初始化"""
        custom_config = {
            'alerts': {
                'enabled': False,
                'levels': {
                    'critical': {'threshold': 0.5, 'channels': ['email']}
                }
            }
        }

        monitor = DataQualityMonitor(custom_config)
        assert monitor.config == custom_config
        assert monitor.alert_config['enabled'] is False


class TestDataQualityMonitorRuleManagement:
    """测试质量规则管理"""

    def setup_method(self):
        """设置测试方法"""
        self.monitor = DataQualityMonitor()

    def test_add_rule(self):
        """测试添加规则"""
        initial_count = len(self.monitor.rules)

        # 创建一个新的规则
        new_rule = CompletenessRule(threshold=0.8)
        self.monitor.add_rule(new_rule)

        assert len(self.monitor.rules) == initial_count + 1
        assert new_rule in self.monitor.rules

    def test_remove_rule(self):
        """测试移除规则"""
        initial_count = len(self.monitor.rules)

        # 移除一个规则
        self.monitor.remove_rule("completeness")
        assert len(self.monitor.rules) == initial_count - 1

        # 确认规则已被移除
        rule_names = [rule.name for rule in self.monitor.rules]
        assert "completeness" not in rule_names

    def test_remove_nonexistent_rule(self):
        """测试移除不存在的规则"""
        initial_count = len(self.monitor.rules)

        # 移除不存在的规则
        self.monitor.remove_rule("NonExistentRule")
        assert len(self.monitor.rules) == initial_count  # 数量不变


class TestDataQualityMonitorQualityCheck:
    """测试质量检查功能"""

    def setup_method(self):
        """设置测试方法"""
        self.monitor = DataQualityMonitor()

    def test_check_quality_perfect_data(self):
        """测试完美数据的质量检查"""
        # 创建完美的数据，使用当前时间附近的时间戳
        now = pd.Timestamp.now()
        perfect_data = pd.DataFrame({
            'price': [100.0, 101.0, 102.0, 103.0, 104.0],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'timestamp': pd.date_range(now - pd.Timedelta(hours=4), periods=5, freq='h')
        })

        report = self.monitor.check_quality(perfect_data, "perfect_data")

        assert report is not None
        assert report.overall_score > 0.9
        assert report.quality_level == QualityLevel.EXCELLENT
        assert report.data_source == "perfect_data"
        assert report.data_shape == (5, 3)

    def test_check_quality_poor_data(self):
        """测试质量差的数据检查"""
        # 创建包含很多缺失值的数据
        now = pd.Timestamp.now()
        poor_data = pd.DataFrame({
            'price': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'volume': [np.nan, np.nan, np.nan, np.nan, 1400],
            'timestamp': pd.date_range(now - pd.Timedelta(hours=4), periods=5, freq='h')
        })

        report = self.monitor.check_quality(poor_data, "poor_data")

        assert report is not None
        assert report.overall_score < 0.9  # 应该较低
        assert report.quality_level in [QualityLevel.GOOD, QualityLevel.POOR, QualityLevel.CRITICAL]
        assert len(report.anomalies) > 0  # 应该有异常

    def test_check_quality_empty_data(self):
        """测试空数据的质量检查"""
        empty_data = pd.DataFrame()

        report = self.monitor.check_quality(empty_data, "empty_data")

        assert report is not None
        assert report.overall_score < 0.6  # 空数据得分应该很低
        assert report.quality_level == QualityLevel.CRITICAL
        assert len(report.anomalies) > 0

    def test_check_quality_with_consistency_issues(self):
        """测试一致性问题的质量检查"""
        # 创建不一致的数据 - 混合数据类型
        now = pd.Timestamp.now()
        inconsistent_data = pd.DataFrame({
            'price': [100.0, "invalid", 102.0, 103.0, 104.0],  # 混合类型
            'volume': [1000, 1100, 1200, 1300, 1400],
            'timestamp': pd.date_range(now - pd.Timedelta(hours=4), periods=5, freq='h')
        })

        report = self.monitor.check_quality(inconsistent_data, "inconsistent_data")

        assert report is not None
        # 应该检测到一致性问题
        assert len(report.anomalies) > 0 or report.metrics['consistency'].value < 1.0


class TestDataQualityRules:
    """测试各种质量规则"""

    def test_completeness_rule_perfect(self):
        """测试完整性规则 - 完美数据"""
        rule = CompletenessRule()
        perfect_data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'w', 'v']
        })

        metric = rule.check(perfect_data)

        assert metric.name == "completeness"
        assert metric.value == 1.0  # 100%完整
        assert metric.status == "excellent"

    def test_completeness_rule_with_missing(self):
        """测试完整性规则 - 包含缺失值"""
        rule = CompletenessRule()
        incomplete_data = pd.DataFrame({
            'a': [1, np.nan, 3, 4, np.nan],
            'b': [10, 20, np.nan, 40, 50],
            'c': ['x', 'y', 'z', 'w', 'v']
        })

        metric = rule.check(incomplete_data)

        assert metric.name == "completeness"
        assert metric.value < 1.0  # 不完整
        assert metric.status != "excellent"

    def test_consistency_rule(self):
        """测试一致性规则"""
        rule = ConsistencyRule()

        # 正常数据
        normal_data = pd.DataFrame({
            'price': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200]
        })

        metric = rule.check(normal_data)

        assert metric.name == "consistency"
        assert metric.value > 0.8  # 应该很高

    def test_accuracy_rule(self):
        """测试准确性规则"""
        rule = AccuracyRule()

        # 包含合理数值的数据
        accurate_data = pd.DataFrame({
            'price': [100.0, 101.0, 102.0, 103.0, 104.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        metric = rule.check(accurate_data)

        assert metric.name == "accuracy"
        assert metric.value > 0.8  # 应该合理

    def test_timeliness_rule(self):
        """测试时效性规则"""
        rule = TimelinessRule()

        # 包含时间戳的数据
        now = pd.Timestamp.now()
        timely_data = pd.DataFrame({
            'price': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200],
            'timestamp': pd.date_range(now - pd.Timedelta(hours=2), periods=3, freq='h')
        })

        metric = rule.check(timely_data)

        assert metric.name == "timeliness"
        assert metric.value > 0.8  # 应该及时


class TestDataQualityMonitorReporting:
    """测试质量监控器报告功能"""

    def setup_method(self):
        """设置测试方法"""
        self.monitor = DataQualityMonitor()

    def test_report_history_storage(self):
        """测试报告历史存储"""
        initial_history_count = len(self.monitor.report_history)

        # 执行质量检查
        test_data = pd.DataFrame({'a': [1, 2, 3]})
        self.monitor.check_quality(test_data, "test_source")

        assert len(self.monitor.report_history) == initial_history_count + 1

    def test_anomaly_history_storage(self):
        """测试异常历史存储"""
        initial_anomaly_count = len(self.monitor.anomaly_history)

        # 执行质量检查 - 使用质量差的数据触发异常
        poor_data = pd.DataFrame({
            'price': [np.nan, np.nan, np.nan],
            'volume': [np.nan, np.nan, np.nan]
        })
        self.monitor.check_quality(poor_data, "poor_source")

        assert len(self.monitor.anomaly_history) > initial_anomaly_count


class TestDataQualityMonitorAlerting:
    """测试质量监控器告警功能"""

    def setup_method(self):
        """设置测试方法"""
        self.monitor = DataQualityMonitor()

    @patch('src.data.quality.data_quality_monitor.logger')
    def test_alert_disabled(self, mock_logger):
        """测试告警禁用"""
        # 禁用告警
        self.monitor.alert_config['enabled'] = False

        poor_data = pd.DataFrame({
            'price': [np.nan, np.nan, np.nan],
            'volume': [np.nan, np.nan, np.nan]
        })

        # 应该不会触发告警
        self.monitor.check_quality(poor_data, "test")

        # 验证没有发送告警
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_determine_quality_level(self):
        """测试质量等级确定"""
        monitor = DataQualityMonitor()

        assert monitor._determine_quality_level(0.95) == QualityLevel.EXCELLENT
        assert monitor._determine_quality_level(0.85) == QualityLevel.GOOD
        assert monitor._determine_quality_level(0.75) == QualityLevel.FAIR
        assert monitor._determine_quality_level(0.65) == QualityLevel.POOR
        assert monitor._determine_quality_level(0.55) == QualityLevel.CRITICAL

    def test_determine_alert_level(self):
        """测试告警级别确定"""
        monitor = DataQualityMonitor()

        assert monitor._determine_alert_level(0.95) is None  # 高分不触发告警
        assert monitor._determine_alert_level(0.85) == AlertLevel.INFO
        assert monitor._determine_alert_level(0.75) == AlertLevel.WARNING
        assert monitor._determine_alert_level(0.65) == AlertLevel.ERROR
        assert monitor._determine_alert_level(0.55) == AlertLevel.CRITICAL  # 低于0.6触发critical


class TestDataQualityMonitorErrorHandling:
    """测试错误处理"""

    def setup_method(self):
        """设置测试方法"""
        self.monitor = DataQualityMonitor()

    def test_rule_execution_failure(self):
        """测试规则执行失败处理"""
        # 创建一个会失败的规则
        class FailingRule(DataQualityRule):
            def __init__(self):
                super().__init__("FailingRule", 1.0)

            def check(self, data):
                raise Exception("Rule execution failed")

        self.monitor.add_rule(FailingRule())

        test_data = pd.DataFrame({'a': [1, 2, 3]})
        report = self.monitor.check_quality(test_data, "test")

        # 应该生成报告，即使有规则失败
        assert report is not None
        assert "FailingRule" in report.metrics
        assert report.metrics["FailingRule"].status == "critical"

    def test_empty_dataframe_handling(self):
        """测试空数据框处理"""
        empty_data = pd.DataFrame()
        report = self.monitor.check_quality(empty_data, "empty")

        assert report is not None
        assert report.overall_score < 0.6  # 空数据得分应该很低
        assert report.data_shape == (0, 0)


class TestDataQualityMonitorIntegration:
    """测试质量监控器集成场景"""

    def setup_method(self):
        """设置测试方法"""
        self.monitor = DataQualityMonitor()

    def test_complete_quality_workflow(self):
        """测试完整质量工作流"""
        # 1. 添加自定义规则
        custom_rule = CompletenessRule(threshold=0.9)
        self.monitor.add_rule(custom_rule)

        # 2. 执行质量检查
        workflow_data = pd.DataFrame({
            'price': [100.0, 101.0, 102.0, np.nan, 104.0],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'timestamp': pd.date_range(pd.Timestamp.now() - pd.Timedelta(hours=4), periods=5, freq='h')
        })

        report = self.monitor.check_quality(workflow_data, "workflow_test")

        # 3. 验证结果
        assert report is not None
        assert report.data_source == "workflow_test"
        assert len(report.metrics) >= 4  # 至少有默认的4个规则
        assert isinstance(report.timestamp, datetime)

        # 4. 验证历史记录
        assert len(self.monitor.report_history) > 0
        assert self.monitor.report_history[-1] == report

    def test_multiple_quality_checks(self):
        """测试多次质量检查"""
        initial_report_count = len(self.monitor.report_history)

        # 执行多次检查
        for i in range(3):
            test_data = pd.DataFrame({
                'price': [100.0 + i, 101.0 + i, 102.0 + i],
                'volume': [1000 + i*10, 1100 + i*10, 1200 + i*10]
            })

            report = self.monitor.check_quality(test_data, f"source_{i}")
            assert report is not None

        # 验证历史记录
        assert len(self.monitor.report_history) == initial_report_count + 3

        # 验证数据源正确
        sources = [r.data_source for r in self.monitor.report_history[-3:]]
        assert sources == ["source_0", "source_1", "source_2"]