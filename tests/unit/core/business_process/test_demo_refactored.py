#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
重构后的BusinessProcessDemo单元测试
测试组合模式拆分后的各个组件职责
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# 尝试导入所需模块
try:
    from src.core.business_process.demo.demo import BusinessProcessDemo
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False

# 使用条件跳过而不是模块级跳过
pytestmark = pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")


class TestDemoConfig:
    """测试演示配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = DemoConfig()
        assert config.symbols == ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        assert config.strategy_name == "demo_strategy"
        assert config.max_processes == 5
        assert config.process_timeout == 300
        assert config.enable_monitoring is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = DemoConfig(
            symbols=["TEST"],
            strategy_name="test_strategy",
            max_processes=3
        )
        assert config.symbols == ["TEST"]
        assert config.strategy_name == "test_strategy"
        assert config.max_processes == 3


class TestDemoMetrics:
    """测试演示指标管理"""

    def test_metrics_initialization(self):
        """测试指标初始化"""
        metrics = DemoMetrics()
        assert metrics.metrics['total_processes'] == 0
        assert metrics.metrics['completed_processes'] == 0
        assert metrics.metrics['failed_processes'] == 0

    def test_metrics_reset(self):
        """测试指标重置"""
        metrics = DemoMetrics()
        metrics.update_metric('total_processes', 5)
        metrics.reset()
        assert metrics.metrics['total_processes'] == 0

    def test_metrics_update(self):
        """测试指标更新"""
        metrics = DemoMetrics()
        metrics.update_metric('total_processes', 10)
        metrics.increment_metric('completed_processes', 5)

        assert metrics.metrics['total_processes'] == 10
        assert metrics.metrics['completed_processes'] == 5

    def test_metrics_summary(self):
        """测试指标摘要"""
        metrics = DemoMetrics()
        summary = metrics.get_summary()
        assert isinstance(summary, dict)
        assert 'total_processes' in summary


class TestDemoInitializer:
    """测试演示初始化器"""

    def test_initializer_creation(self):
        """测试初始化器创建"""
        config = DemoConfig()
        initializer = DemoInitializer(config)
        assert initializer.config == config
        assert initializer.event_bus_initializer is not None

    @patch('core.business_process.examples.demo.EventBus')
    @patch('core.business_process.examples.demo.DependencyContainer')
    @patch('core.business_process.examples.demo.ServiceContainer')
    @patch('core.business_process.examples.demo.BusinessProcessOrchestrator')
    @patch('core.business_process.examples.demo.BusinessProcessIntegration')
    def test_initialize_all_components_success(self, mock_integration, mock_orchestrator,
                                             mock_service_container, mock_container, mock_event_bus):
        """测试组件初始化成功"""
        # 配置mock对象
        mock_event_bus.return_value.initialize.return_value = True
        mock_container.return_value.initialize.return_value = True
        mock_orchestrator.return_value.initialize.return_value = True
        mock_integration.return_value.initialize.return_value = True

        config = DemoConfig()
        initializer = DemoInitializer(config)

        result = initializer.initialize_all_components()

        assert result is True
        mock_event_bus.return_value.initialize.assert_called_once()
        mock_container.return_value.initialize.assert_called_once()
        mock_orchestrator.return_value.initialize.assert_called_once()
        mock_integration.return_value.initialize.assert_called_once()

    def test_get_components(self):
        """测试获取组件"""
        config = DemoConfig()
        initializer = DemoInitializer(config)

        components = initializer.get_components()

        assert isinstance(components, dict)
        assert 'event_bus' in components
        assert 'container' in components
        assert 'orchestrator' in components


class TestDemoMonitor:
    """测试演示监控器"""

    @patch('core.business_process.examples.demo.BusinessProcessOrchestrator')
    @patch('core.business_process.examples.demo.EventBus')
    def test_monitor_creation(self, mock_event_bus, mock_orchestrator):
        """测试监控器创建"""
        metrics = DemoMetrics()
        monitor = DemoMonitor(metrics, mock_orchestrator, mock_event_bus)

        assert monitor.metrics == metrics
        assert monitor.orchestrator == mock_orchestrator
        assert monitor.event_bus == mock_event_bus

    @patch('core.business_process.examples.demo.BusinessProcessOrchestrator')
    @patch('core.business_process.examples.demo.EventBus')
    def test_get_demo_status(self, mock_event_bus, mock_orchestrator):
        """测试获取演示状态"""
        mock_orchestrator.get_current_state.return_value.value = 'running'
        mock_orchestrator.get_running_processes.return_value = ['process1', 'process2']
        mock_event_bus.get_event_statistics.return_value = {'events': 10}

        metrics = DemoMetrics()
        monitor = DemoMonitor(metrics, mock_orchestrator, mock_event_bus)

        status = monitor.get_demo_status()

        assert status['is_running'] is True
        assert status['orchestrator_status'] == 'running'
        assert status['running_processes'] == 2
        assert status['event_bus_stats'] == {'events': 10}

    @patch('core.business_process.examples.demo.BusinessProcessOrchestrator')
    @patch('core.business_process.examples.demo.EventBus')
    def test_get_demo_metrics(self, mock_event_bus, mock_orchestrator):
        """测试获取演示指标"""
        metrics = DemoMetrics()
        metrics.update_metric('total_processes', 10)
        metrics.update_metric('completed_processes', 8)
        metrics.update_metric('failed_processes', 2)  # 添加失败进程数

        monitor = DemoMonitor(metrics, mock_orchestrator, mock_event_bus)

        result = monitor.get_demo_metrics()

        assert result['total_processes'] == 10
        assert result['completed_processes'] == 8
        assert result['failed_processes'] == 2
        assert result['success_rate'] == 80.0
        assert result['failure_rate'] == 20.0


class TestDemoEventHandler:
    """测试演示事件处理器"""

    @patch('core.business_process.examples.demo.BusinessProcessOrchestrator')
    @patch('core.business_process.examples.demo.EventBus')
    def test_event_handler_creation(self, mock_event_bus, mock_orchestrator):
        """测试事件处理器创建"""
        metrics = DemoMetrics()
        handler = DemoEventHandler(mock_event_bus, mock_orchestrator, metrics)

        assert handler.event_bus == mock_event_bus
        assert handler.orchestrator == mock_orchestrator
        assert handler.metrics == metrics

    @patch('core.business_process.examples.demo.BusinessProcessOrchestrator')
    @patch('core.business_process.examples.demo.EventBus')
    def test_setup_event_handlers(self, mock_event_bus, mock_orchestrator):
        """测试事件处理器设置"""
        metrics = DemoMetrics()
        handler = DemoEventHandler(mock_event_bus, mock_orchestrator, metrics)

        handler.setup_event_handlers()

        # 验证所有事件处理器都被订阅
        assert mock_event_bus.subscribe.call_count == 10  # 10种不同的事件类型

    @patch('core.business_process.examples.demo.BusinessProcessOrchestrator')
    @patch('core.business_process.examples.demo.EventBus')
    def test_event_processing(self, mock_event_bus, mock_orchestrator):
        """测试事件处理"""
        metrics = DemoMetrics()
        handler = DemoEventHandler(mock_event_bus, mock_orchestrator, metrics)

        # 模拟进程完成事件
        mock_event = Mock()
        mock_event.data = {'process_id': 'test_process'}

        handler._on_process_completed(mock_event)

        assert metrics.metrics['completed_processes'] == 1
        assert metrics.metrics['active_processes'] == -1


class TestDemoRunner:
    """测试演示运行器"""

    @patch('core.business_process.examples.demo.DemoEventHandler')
    @patch('core.business_process.examples.demo.DemoInitializer')
    @patch('core.business_process.examples.demo.DemoMetrics')
    def test_runner_creation(self, mock_metrics, mock_initializer, mock_event_handler):
        """测试运行器创建"""
        config = DemoConfig()
        runner = DemoRunner(config, mock_initializer, mock_event_handler, mock_metrics)

        assert runner.config == config
        assert runner.initializer == mock_initializer
        assert runner.event_handler == mock_event_handler
        assert runner.metrics == mock_metrics
        assert runner.is_running is False
        assert runner.demo_processes == []

    @patch('core.business_process.examples.demo.DemoEventHandler')
    @patch('core.business_process.examples.demo.DemoInitializer')
    @patch('core.business_process.examples.demo.DemoMetrics')
    @patch('core.business_process.examples.demo.BusinessProcessOrchestrator')
    @patch('core.business_process.examples.demo.EventBus')
    def test_start_demo_success(self, mock_event_bus, mock_orchestrator, mock_metrics, mock_initializer, mock_event_handler):
        """测试演示启动成功"""
        config = DemoConfig(symbols=["TEST"])
        mock_event_bus_instance = Mock()
        mock_initializer.get_components.return_value = {
            'orchestrator': mock_orchestrator,
            'event_bus': mock_event_bus_instance
        }

        runner = DemoRunner(config, mock_initializer, mock_event_handler, mock_metrics)

        result = runner.start_demo()

        assert result is True
        assert runner.is_running is True
        assert len(runner.demo_processes) == 1
        mock_metrics.reset.assert_called_once()

    @patch('core.business_process.examples.demo.DemoEventHandler')
    @patch('core.business_process.examples.demo.DemoInitializer')
    @patch('core.business_process.examples.demo.DemoMetrics')
    @patch('core.business_process.examples.demo.BusinessProcessOrchestrator')
    def test_stop_demo_success(self, mock_orchestrator, mock_metrics, mock_initializer, mock_event_handler):
        """测试演示停止成功"""
        config = DemoConfig()
        mock_initializer.get_components.return_value = {
            'orchestrator': mock_orchestrator,
            'integration': Mock(),
            'event_bus': Mock(),
            'container': Mock()
        }

        runner = DemoRunner(config, mock_initializer, mock_event_handler, mock_metrics)
        runner.is_running = True
        runner.demo_processes = ['process1']

        result = runner.stop_demo()

        assert result is True
        assert runner.is_running is False
        mock_orchestrator.pause_process.assert_called_once_with('process1')


class TestBusinessProcessDemo:
    """测试业务流程演示类"""

    def test_demo_creation(self):
        """测试演示创建"""
        config = DemoConfig()
        demo = BusinessProcessDemo(config)

        assert demo.config == config
        assert demo.metrics is not None
        assert demo.initializer is not None
        assert demo.event_handler is None  # 初始为None，在初始化后创建
        assert demo.runner is None
        assert demo.monitor is None

    @patch('core.business_process.examples.demo.DemoInitializer')
    @patch('core.business_process.examples.demo.DemoEventHandler')
    @patch('core.business_process.examples.demo.DemoRunner')
    @patch('core.business_process.examples.demo.DemoMonitor')
    def test_initialize_success(self, mock_monitor, mock_runner, mock_event_handler, mock_initializer):
        """测试初始化成功"""
        config = DemoConfig()
        demo = BusinessProcessDemo(config)

        mock_initializer.return_value.initialize_all_components.return_value = True
        mock_initializer.return_value.get_components.return_value = {
            'event_bus': Mock(),
            'container': Mock(),
            'orchestrator': Mock()
        }

        result = demo.initialize()

        assert result is True
        assert demo.event_handler is not None
        assert demo.runner is not None
        assert demo.monitor is not None

    @patch('core.business_process.examples.demo.DemoRunner')
    def test_start_demo_delegation(self, mock_runner_class):
        """测试启动演示委派"""
        config = DemoConfig()
        demo = BusinessProcessDemo(config)
        demo.runner = Mock()

        demo.start_demo()

        demo.runner.start_demo.assert_called_once()

    @patch('core.business_process.examples.demo.DemoRunner')
    def test_stop_demo_delegation(self, mock_runner_class):
        """测试停止演示委派"""
        config = DemoConfig()
        demo = BusinessProcessDemo(config)
        demo.runner = Mock()

        demo.stop_demo()

        demo.runner.stop_demo.assert_called_once()

    @patch('core.business_process.examples.demo.DemoMonitor')
    @patch('core.business_process.examples.demo.DemoRunner')
    def test_get_demo_status_delegation(self, mock_runner, mock_monitor):
        """测试获取演示状态委派"""
        config = DemoConfig()
        demo = BusinessProcessDemo(config)
        demo.monitor = Mock()
        demo.runner = Mock()
        demo.monitor.get_demo_status.return_value = {'is_running': True}
        demo.runner.demo_processes = ['process1']

        status = demo.get_demo_status()

        assert status['is_running'] is True
        assert status['process_list'] == ['process1']

    @patch('core.business_process.examples.demo.DemoMonitor')
    def test_get_demo_metrics_delegation(self, mock_monitor):
        """测试获取演示指标委派"""
        config = DemoConfig()
        demo = BusinessProcessDemo(config)
        demo.monitor = Mock()
        demo.monitor.get_demo_metrics.return_value = {'total_processes': 10}

        metrics = demo.get_demo_metrics()

        assert metrics['total_processes'] == 10


if __name__ == "__main__":
    pytest.main([__file__])
