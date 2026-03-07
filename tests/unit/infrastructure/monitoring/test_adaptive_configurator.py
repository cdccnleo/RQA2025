#!/usr/bin/env python3
"""
RQA2025 基础设施层自适应配置器单元测试
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.infrastructure.monitoring.components.adaptive_configurator import (
    AdaptiveConfigurator, ConfigurationRule, AdaptationHistory,
    AdaptationStrategy, create_adaptive_configurator
)


class TestConfigurationRule:
    """测试配置规则"""

    def test_initialization(self):
        """测试初始化"""
        rule = ConfigurationRule(
            parameter_path="monitoring.interval",
            metric_name="cpu_usage",
            condition="cpu_usage > 80",
            action="increase_interval",
            priority=5,
            cooldown_minutes=10
        )

        assert rule.parameter_path == "monitoring.interval"
        assert rule.metric_name == "cpu_usage"
        assert rule.condition == "cpu_usage > 80"
        assert rule.action == "increase_interval"
        assert rule.priority == 5
        assert rule.cooldown_minutes == 10
        assert rule.last_applied is None


class TestAdaptiveConfigurator:
    """测试自适应配置器"""

    def setup_method(self):
        """测试前准备"""
        self.configurator = AdaptiveConfigurator(
            strategy=AdaptationStrategy.BALANCED,
            load_default_rules=False,
        )

    def teardown_method(self):
        """测试后清理"""
        if self.configurator.is_running:
            self.configurator.stop()

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_performance_monitor')
    def test_initialization(self, mock_monitor):
        """测试初始化"""
        assert self.configurator.strategy == AdaptationStrategy.BALANCED
        assert self.configurator.monitoring_interval == 60
        assert not self.configurator.is_running
        assert len(self.configurator.rules) == 0
        assert len(self.configurator.adaptation_history) == 0

    def test_add_remove_rule(self):
        """测试添加和移除规则"""
        rule = ConfigurationRule(
            parameter_path="test.param",
            metric_name="cpu_usage",
            condition="cpu_usage > 80",
            action="increase_interval"
        )

        # 添加规则
        self.configurator.add_rule(rule)
        assert len(self.configurator.rules) == 1
        assert self.configurator.rules[0].parameter_path == "test.param"

        # 移除规则
        self.configurator.remove_rule("test.param")
        assert len(self.configurator.rules) == 0

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_performance_monitor')
    def test_start_stop(self, mock_monitor):
        """测试启动和停止"""
        mock_monitor.get_recent_metrics.return_value = {}

        # 启动
        self.configurator.start()
        assert self.configurator.is_running
        assert self.configurator.monitor_thread.is_alive()

        # 停止
        self.configurator.stop()
        assert not self.configurator.is_running
        assert not (self.configurator.monitor_thread.is_alive() if self.configurator.monitor_thread else False)

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_performance_monitor')
    def test_evaluate_condition(self, mock_monitor):
        """测试条件评估"""
        mock_monitor.get_recent_metrics.return_value = {'cpu_usage': 85}

        # 测试大于条件
        assert self.configurator._evaluate_condition("cpu_usage > 80")
        assert not self.configurator._evaluate_condition("cpu_usage > 90")

        # 测试小于条件
        assert self.configurator._evaluate_condition("cpu_usage < 90")
        assert not self.configurator._evaluate_condition("cpu_usage < 80")

        # 测试等于条件
        mock_monitor.get_recent_metrics.return_value = {'cpu_usage': 80}
        assert self.configurator._evaluate_condition("cpu_usage == 80")
        assert not self.configurator._evaluate_condition("cpu_usage == 85")

        # 测试无效条件
        assert not self.configurator._evaluate_condition("invalid condition")
        assert not self.configurator._evaluate_condition("nonexistent_metric > 50")

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_performance_monitor')
    def test_calculate_new_value(self, mock_monitor):
        """测试新值计算"""
        # 测试增加动作
        current_value = 60
        new_value = self.configurator._calculate_new_value("increase_interval", current_value, "test.param")
        assert new_value == current_value * 1.25  # BALANCED 策略

        # 测试减少动作
        new_value = self.configurator._calculate_new_value("decrease_interval", current_value, "test.param")
        assert new_value == current_value / 1.25

        # 测试无效动作
        new_value = self.configurator._calculate_new_value("invalid_action", current_value, "test.param")
        assert new_value == current_value

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_performance_monitor')
    def test_conservative_strategy(self, mock_monitor):
        """测试保守策略"""
        configurator = AdaptiveConfigurator(strategy=AdaptationStrategy.CONSERVATIVE)
        current_value = 100

        new_value = configurator._calculate_new_value("increase_interval", current_value, "test.param")
        assert new_value == current_value * 1.1  # 保守策略

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_performance_monitor')
    def test_aggressive_strategy(self, mock_monitor):
        """测试激进策略"""
        configurator = AdaptiveConfigurator(strategy=AdaptationStrategy.AGGRESSIVE)
        current_value = 100

        new_value = configurator._calculate_new_value("increase_interval", current_value, "test.param")
        assert new_value == current_value * 1.5  # 激进策略

    def test_get_baseline_stats(self):
        """测试获取基线统计"""
        # 添加一些基线数据
        with self.configurator.baseline_lock:
            self.configurator.baseline_data['cpu_usage'] = [50, 60, 70, 80, 90]

        stats = self.configurator.get_baseline_stats('cpu_usage')
        assert stats is not None
        assert stats['mean'] == 70.0
        assert stats['min'] == 50
        assert stats['max'] == 90
        assert stats['count'] == 5

        # 测试不存在的指标
        stats = self.configurator.get_baseline_stats('nonexistent')
        assert stats is None

    def test_get_adaptation_history(self):
        """测试获取适应历史"""
        # 添加一些历史记录
        history1 = AdaptationHistory(
            timestamp=datetime.now() - timedelta(hours=1),
            parameter_path="test.param1",
            old_value=60,
            new_value=75,
            reason="cpu_high"
        )
        history2 = AdaptationHistory(
            timestamp=datetime.now(),
            parameter_path="test.param2",
            old_value=100,
            new_value=120,
            reason="memory_high"
        )

        with self.configurator.history_lock:
            self.configurator.adaptation_history = [history1, history2]

        # 获取所有历史
        history = self.configurator.get_adaptation_history()
        assert len(history) == 2
        assert history[0].parameter_path == "test.param2"  # 最新的在前

        # 按参数过滤
        history = self.configurator.get_adaptation_history("test.param1")
        assert len(history) == 1
        assert history[0].parameter_path == "test.param1"

    def test_emergency_adaptation(self):
        """测试紧急适应"""
        initial_rules_count = len(self.configurator.rules)

        self.configurator._emergency_adaptation("high_cpu_load")

        # 应该添加了紧急规则
        assert len(self.configurator.rules) > initial_rules_count

    def test_get_health_status(self):
        """测试获取健康状态"""
        # 添加一些测试数据
        rule = ConfigurationRule(
            parameter_path="test.param",
            metric_name="cpu_usage",
            condition="cpu_usage > 80",
            action="increase_interval"
        )
        self.configurator.add_rule(rule)

        history = AdaptationHistory(
            timestamp=datetime.now(),
            parameter_path="test.param",
            old_value=60,
            new_value=75,
            reason="test"
        )
        with self.configurator.history_lock:
            self.configurator.adaptation_history.append(history)

        with self.configurator.baseline_lock:
            self.configurator.baseline_data['cpu_usage'] = [50, 60, 70]

        status = self.configurator.get_health_status()
        assert status['status'] == 'stopped'
        assert status['rules_count'] == 1
        assert status['history_count'] == 1
        assert status['baseline_metrics'] == 1
        assert status['strategy'] == 'balanced'

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_component_bus')
    def test_handle_performance_update(self, mock_bus):
        """测试处理性能更新事件"""
        message = Mock()
        self.configurator._handle_performance_update(message)
        # 应该没有异常抛出
        assert True

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_component_bus')
    def test_handle_load_change(self, mock_bus):
        """测试处理负载变化事件"""
        message = Mock()
        message.payload = {'cpu_usage': 95}

        with patch.object(self.configurator, '_emergency_adaptation') as mock_emergency:
            self.configurator._handle_load_change(message)
            mock_emergency.assert_called_once_with("high_cpu_load")

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_component_bus')
    def test_handle_config_change(self, mock_bus):
        """测试处理配置变更事件"""
        message = Mock()
        message.payload = {
            'component': 'test_component',
            'key': 'test_key',
            'new_value': 123
        }

        self.configurator._handle_config_change(message)

        with self.configurator.snapshot_lock:
            assert 'test_component' in self.configurator.config_snapshots
            assert 'test_key' in self.configurator.config_snapshots['test_component']
            snapshot = self.configurator.config_snapshots['test_component']['test_key']
            assert snapshot['value'] == 123


class TestCreateAdaptiveConfigurator:
    """测试创建自适应配置器函数"""

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_performance_monitor')
    def test_create_with_defaults(self, mock_monitor):
        """测试使用默认参数创建"""
        configurator = create_adaptive_configurator()

        assert configurator.strategy == AdaptationStrategy.BALANCED
        # 应该有默认规则
        assert len(configurator.rules) >= 3  # DEFAULT_ADAPTATION_RULES 中的规则数量

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_performance_monitor')
    def test_create_with_custom_rules(self, mock_monitor):
        """测试使用自定义规则创建"""
        custom_rule = ConfigurationRule(
            parameter_path="custom.param",
            metric_name="memory_usage",
            condition="memory_usage > 90",
            action="optimize_memory"
        )

        configurator = create_adaptive_configurator(rules=[custom_rule])

        # 应该包含默认规则和自定义规则
        assert len(configurator.rules) >= 4  # 3个默认 + 1个自定义
        assert any(r.parameter_path == "custom.param" for r in configurator.rules)

    @patch('src.infrastructure.monitoring.components.adaptive_configurator.global_performance_monitor')
    def test_create_with_strategy(self, mock_monitor):
        """测试使用指定策略创建"""
        configurator = create_adaptive_configurator(strategy=AdaptationStrategy.CONSERVATIVE)

        assert configurator.strategy == AdaptationStrategy.CONSERVATIVE
