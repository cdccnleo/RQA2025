"""
执行监控器单元测试

测试ExecutionMonitor的核心功能
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.gateway.web.execution_monitor import (
    ExecutionMonitor, ExecutionStatus, ExecutionMetrics,
    MonitoringRule, get_execution_monitor
)


class TestExecutionStatus:
    """测试执行状态枚举"""
    
    def test_status_values(self):
        """测试状态值定义"""
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.PAUSED.value == "paused"
        assert ExecutionStatus.STOPPED.value == "stopped"
        assert ExecutionStatus.ERROR.value == "error"
        assert ExecutionStatus.STARTING.value == "starting"
        assert ExecutionStatus.STOPPING.value == "stopping"


class TestExecutionMetrics:
    """测试执行指标类"""
    
    def test_initialization(self):
        """测试指标初始化"""
        metrics = ExecutionMetrics(strategy_id="test_001")
        
        assert metrics.strategy_id == "test_001"
        assert metrics.latency_ms == 0.0
        assert metrics.throughput == 0.0
        assert metrics.signal_count == 0
        assert metrics.error_count == 0
        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0


class TestMonitoringRule:
    """测试监控规则类"""
    
    def test_initialization(self):
        """测试规则初始化"""
        rule = MonitoringRule(
            rule_id="test_rule",
            rule_type="latency",
            threshold=1000.0,
            duration=60,
            severity="warning"
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.rule_type == "latency"
        assert rule.threshold == 1000.0
        assert rule.duration == 60
        assert rule.severity == "warning"
        assert rule.enabled is True


class TestExecutionMonitor:
    """测试执行监控器"""
    
    @pytest.fixture
    def monitor(self):
        """创建监控器实例"""
        return ExecutionMonitor()
    
    def test_singleton(self):
        """测试单例模式"""
        monitor1 = get_execution_monitor()
        monitor2 = get_execution_monitor()
        assert monitor1 is monitor2
        
    def test_initialization(self, monitor):
        """测试监控器初始化"""
        assert monitor._running is False
        assert len(monitor._monitored_strategies) == 0
        assert len(monitor._metrics_cache) == 0
        assert len(monitor._monitoring_rules) > 0  # 应该有默认规则
        
    def test_register_strategy(self, monitor):
        """测试注册策略"""
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="测试策略"
        )
        
        assert "test_001" in monitor._monitored_strategies
        assert monitor._monitored_strategies["test_001"]["strategy_name"] == "测试策略"
        assert monitor._monitored_strategies["test_001"]["status"] == ExecutionStatus.STOPPED
        
    def test_unregister_strategy(self, monitor):
        """测试注销策略"""
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="测试策略"
        )
        monitor.unregister_strategy("test_001")
        
        assert "test_001" not in monitor._monitored_strategies
        
    def test_update_strategy_status(self, monitor):
        """测试更新策略状态"""
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="测试策略"
        )
        monitor.update_strategy_status(
            strategy_id="test_001",
            status=ExecutionStatus.RUNNING,
            reason="测试启动"
        )
        
        assert monitor._monitored_strategies["test_001"]["status"] == ExecutionStatus.RUNNING
        
    def test_update_metrics(self, monitor):
        """测试更新指标"""
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="测试策略"
        )
        
        monitor.update_metrics(
            strategy_id="test_001",
            latency_ms=100.0,
            throughput=10.0,
            signal_count=5,
            error_count=0
        )
        
        assert "test_001" in monitor._metrics_cache
        cached_metrics = monitor._metrics_cache["test_001"]
        assert cached_metrics.latency_ms == 100.0
        assert cached_metrics.throughput == 10.0
        assert cached_metrics.signal_count == 5
        
    def test_get_execution_status(self, monitor):
        """测试获取策略执行状态"""
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="测试策略"
        )
        monitor.update_strategy_status(
            strategy_id="test_001",
            status=ExecutionStatus.RUNNING
        )
        
        status = monitor.get_execution_status("test_001")
        assert status is not None
        assert status == ExecutionStatus.RUNNING
        
        # 获取不存在的策略
        status2 = monitor.get_execution_status("nonexistent")
        assert status2 is None
        
    def test_get_metrics(self, monitor):
        """测试获取策略指标"""
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="测试策略"
        )
        
        metrics = monitor.get_metrics("test_001")
        assert metrics is not None
        assert metrics.strategy_id == "test_001"
        
    def test_get_all_metrics(self, monitor):
        """测试获取所有策略指标"""
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="测试策略1"
        )
        monitor.register_strategy(
            strategy_id="test_002",
            strategy_name="测试策略2"
        )
        
        all_metrics = monitor.get_all_metrics()
        assert len(all_metrics) == 2
        assert "test_001" in all_metrics
        assert "test_002" in all_metrics
        
    def test_add_monitoring_rule(self, monitor):
        """测试添加监控规则"""
        rule = MonitoringRule(
            rule_id="custom_rule",
            rule_type="custom",
            threshold=500.0,
            duration=30,
            severity="high"
        )
        
        monitor.add_monitoring_rule(rule)
        assert "custom_rule" in monitor._monitoring_rules
        
    def test_remove_monitoring_rule(self, monitor):
        """测试移除监控规则"""
        rule = MonitoringRule(
            rule_id="temp_rule",
            rule_type="temp",
            threshold=100.0,
            duration=10,
            severity="low"
        )
        monitor.add_monitoring_rule(rule)
        monitor.remove_monitoring_rule("temp_rule")
        
        assert "temp_rule" not in monitor._monitoring_rules
        
    def test_add_status_callback(self, monitor):
        """测试添加状态回调"""
        callback_called = False
        
        def test_callback(strategy_id, status):
            nonlocal callback_called
            callback_called = True
            
        monitor.add_status_callback(test_callback)
        assert test_callback in monitor._status_callbacks
        
    def test_add_metrics_callback(self, monitor):
        """测试添加指标回调"""
        def test_callback(metrics):
            pass
            
        monitor.add_metrics_callback(test_callback)
        assert test_callback in monitor._metrics_callbacks
        
    def test_enable_disable_rule(self, monitor):
        """测试启用/禁用规则"""
        # 获取第一个默认规则
        rule_id = list(monitor._monitoring_rules.keys())[0]
        
        # 禁用规则
        monitor.disable_rule(rule_id)
        assert monitor._monitoring_rules[rule_id].enabled is False
        
        # 启用规则
        monitor.enable_rule(rule_id)
        assert monitor._monitoring_rules[rule_id].enabled is True
        
    def test_get_metrics_history(self, monitor):
        """测试获取指标历史"""
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="测试策略"
        )
        
        # 更新几次指标以生成历史
        for i in range(3):
            monitor.update_metrics(
                strategy_id="test_001",
                latency_ms=100.0 + i * 10,
                signal_count=i + 1
            )
        
        history = monitor.get_metrics_history("test_001")
        assert isinstance(history, list)
        assert len(history) == 3
        
    def test_get_monitoring_summary(self, monitor):
        """测试获取监控摘要"""
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="测试策略"
        )
        monitor.update_strategy_status(
            strategy_id="test_001",
            status=ExecutionStatus.RUNNING
        )
        
        summary = monitor.get_monitoring_summary()
        assert summary["total_strategies"] == 1
        assert summary["running_count"] == 1
        assert summary["error_count"] == 0
        assert summary["is_running"] is False


class TestExecutionMonitorEdgeCases:
    """测试边界情况"""
    
    @pytest.fixture
    def monitor(self):
        return ExecutionMonitor()
    
    def test_update_metrics_nonexistent_strategy(self, monitor):
        """测试更新不存在的策略指标"""
        # 不应该抛出异常
        monitor.update_metrics(
            strategy_id="nonexistent",
            latency_ms=100.0,
            throughput=10.0,
            signal_count=1,
            error_count=0
        )
        # 未注册的策略不会被添加到缓存
        assert "nonexistent" not in monitor._metrics_cache
        
    def test_update_status_nonexistent_strategy(self, monitor):
        """测试更新不存在的策略状态"""
        # 不应该抛出异常，但也不会添加新策略
        monitor.update_strategy_status(
            strategy_id="nonexistent",
            status=ExecutionStatus.RUNNING
        )
        assert "nonexistent" not in monitor._monitored_strategies
        
    def test_unregister_nonexistent_strategy(self, monitor):
        """测试注销不存在的策略"""
        # 不应该抛出异常
        monitor.unregister_strategy("nonexistent")
        
    def test_enable_nonexistent_rule(self, monitor):
        """测试启用不存在的规则"""
        # 不应该抛出异常
        monitor.enable_rule("nonexistent")
        
    def test_disable_nonexistent_rule(self, monitor):
        """测试禁用不存在的规则"""
        # 不应该抛出异常
        monitor.disable_rule("nonexistent")
        
    def test_remove_nonexistent_rule(self, monitor):
        """测试移除不存在的规则"""
        # 不应该抛出异常
        monitor.remove_monitoring_rule("nonexistent")
        
    def test_metrics_with_zero_values(self, monitor):
        """测试零值指标"""
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="测试策略"
        )
        
        monitor.update_metrics(
            strategy_id="test_001",
            latency_ms=0.0,
            throughput=0.0,
            signal_count=0,
            error_count=0
        )
        
        metrics = monitor.get_metrics("test_001")
        assert metrics.latency_ms == 0.0
        assert metrics.throughput == 0.0
        assert metrics.signal_count == 0
        
    def test_register_duplicate_strategy(self, monitor):
        """测试重复注册策略"""
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="测试策略"
        )
        
        # 重复注册不应该抛出异常
        monitor.register_strategy(
            strategy_id="test_001",
            strategy_name="新名称"
        )
        
        # 名称应该保持不变（第一次注册的）
        assert monitor._monitored_strategies["test_001"]["strategy_name"] == "测试策略"


class TestExecutionMonitorAsync:
    """测试异步方法"""
    
    @pytest.fixture
    def monitor(self):
        return ExecutionMonitor()
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor):
        """测试启动和停止监控"""
        # 启动监控
        await monitor.start_monitoring()
        assert monitor._running is True
        assert monitor._check_task is not None
        
        # 停止监控
        await monitor.stop_monitoring()
        assert monitor._running is False
        
    @pytest.mark.asyncio
    async def test_double_start(self, monitor):
        """测试重复启动"""
        await monitor.start_monitoring()
        first_task = monitor._check_task
        
        # 重复启动不应该创建新任务
        await monitor.start_monitoring()
        assert monitor._check_task is first_task
        
        await monitor.stop_monitoring()
        
    @pytest.mark.asyncio
    async def test_double_stop(self, monitor):
        """测试重复停止"""
        await monitor.start_monitoring()
        await monitor.stop_monitoring()
        
        # 重复停止不应该抛出异常
        await monitor.stop_monitoring()
        assert monitor._running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
