#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试监控协调器组件
"""

import importlib
import sys
import threading
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pytest


@pytest.fixture
def monitoring_coordinator_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.monitoring_coordinator"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def mock_config():
    """创建模拟配置对象"""
    class MockMonitoringConfig:
        def __init__(self):
            self.collection_interval = 1.0
    
    return MockMonitoringConfig()


@pytest.fixture
def coordinator(monitoring_coordinator_module, mock_config):
    """创建MonitoringCoordinator实例"""
    module = monitoring_coordinator_module
    
    # Mock全局依赖
    with patch.object(module, 'global_component_bus', Mock()):
        with patch.object(module, 'monitor_performance', lambda component, operation: lambda func: func):
            coordinator = module.MonitoringCoordinator(
                pool_name="test_pool",
                config=mock_config
            )
            return coordinator, module


def test_initialization(coordinator):
    """测试初始化"""
    coordinator_instance, module = coordinator
    assert coordinator_instance.pool_name == "test_pool"
    assert coordinator_instance.monitoring_active is False
    assert coordinator_instance.monitoring_thread is None
    assert coordinator_instance.start_time is None
    assert coordinator_instance.stats_collector is None
    assert coordinator_instance.alert_manager is None
    assert coordinator_instance.metrics_exporter is None


def test_set_components(coordinator):
    """测试设置组件"""
    coordinator_instance, module = coordinator
    
    mock_stats_collector = Mock()
    mock_alert_manager = Mock()
    mock_metrics_exporter = Mock()
    
    coordinator_instance.set_components(
        mock_stats_collector,
        mock_alert_manager,
        mock_metrics_exporter
    )
    
    assert coordinator_instance.stats_collector == mock_stats_collector
    assert coordinator_instance.alert_manager == mock_alert_manager
    assert coordinator_instance.metrics_exporter == mock_metrics_exporter


def test_start_monitoring_success(coordinator):
    """测试启动监控（成功）"""
    coordinator_instance, module = coordinator
    
    # Mock threading.Thread
    mock_thread = Mock()
    mock_thread.is_alive.return_value = True
    mock_thread.start = Mock()
    
    with patch.object(module.threading, 'Thread', return_value=mock_thread):
        result = coordinator_instance.start_monitoring()
        
        assert result is True
        assert coordinator_instance.monitoring_active is True
        assert coordinator_instance.start_time is not None
        assert coordinator_instance.monitoring_thread == mock_thread
        mock_thread.start.assert_called_once()


def test_start_monitoring_already_active(coordinator):
    """测试启动监控（已经激活）"""
    coordinator_instance, module = coordinator
    
    coordinator_instance.monitoring_active = True
    
    result = coordinator_instance.start_monitoring()
    
    assert result is True


def test_start_monitoring_exception(coordinator):
    """测试启动监控（异常）"""
    coordinator_instance, module = coordinator
    
    # Mock threading.Thread抛出异常
    def failing_thread(*args, **kwargs):
        raise RuntimeError("Thread error")
    
    with patch.object(module.threading, 'Thread', side_effect=failing_thread):
        result = coordinator_instance.start_monitoring()
        
        assert result is False
        assert coordinator_instance.monitoring_active is False


def test_stop_monitoring_success(coordinator):
    """测试停止监控（成功）"""
    coordinator_instance, module = coordinator
    
    # 先启动监控
    mock_thread = Mock()
    mock_thread.is_alive.return_value = True
    mock_thread.join = Mock()
    
    with patch.object(module.threading, 'Thread', return_value=mock_thread):
        coordinator_instance.start_monitoring()
        coordinator_instance.monitoring_active = True
        coordinator_instance.monitoring_thread = mock_thread
        
        result = coordinator_instance.stop_monitoring()
        
        assert result is True
        assert coordinator_instance.monitoring_active is False
        mock_thread.join.assert_called_once_with(timeout=5.0)


def test_stop_monitoring_not_active(coordinator):
    """测试停止监控（未激活）"""
    coordinator_instance, module = coordinator
    
    coordinator_instance.monitoring_active = False
    
    result = coordinator_instance.stop_monitoring()
    
    assert result is True


def test_stop_monitoring_exception(coordinator):
    """测试停止监控（异常）"""
    coordinator_instance, module = coordinator
    
    coordinator_instance.monitoring_active = True
    mock_thread = Mock()
    mock_thread.join.side_effect = RuntimeError("Join error")
    coordinator_instance.monitoring_thread = mock_thread
    
    result = coordinator_instance.stop_monitoring()
    
    assert result is False


def test_get_monitoring_status(coordinator):
    """测试获取监控状态"""
    coordinator_instance, module = coordinator
    
    status = coordinator_instance.get_monitoring_status()
    
    assert status['active'] is False
    assert status['pool_name'] == "test_pool"
    assert status['start_time'] is None
    assert status['uptime_seconds'] == 0
    assert status['collection_interval'] == 1.0


def test_get_monitoring_status_with_start_time(coordinator):
    """测试获取监控状态（有启动时间）"""
    coordinator_instance, module = coordinator
    
    coordinator_instance.start_time = datetime.now()
    coordinator_instance.monitoring_active = True
    
    status = coordinator_instance.get_monitoring_status()
    
    assert status['active'] is True
    assert status['start_time'] is not None
    assert status['uptime_seconds'] >= 0


def test_collect_statistics_with_collector(coordinator):
    """测试收集统计信息（有收集器）"""
    coordinator_instance, module = coordinator
    
    mock_stats_collector = Mock()
    mock_stats_collector.collect_stats.return_value = {'pool_size': 10}
    coordinator_instance.stats_collector = mock_stats_collector
    
    with patch.object(coordinator_instance, '_publish_stats_collected_event', Mock()):
        stats = coordinator_instance._collect_statistics()
        
        assert stats == {'pool_size': 10}
        mock_stats_collector.collect_stats.assert_called_once()


def test_collect_statistics_no_collector(coordinator):
    """测试收集统计信息（无收集器）"""
    coordinator_instance, module = coordinator
    
    stats = coordinator_instance._collect_statistics()
    
    assert stats is None


def test_collect_statistics_none_result(coordinator):
    """测试收集统计信息（返回None）"""
    coordinator_instance, module = coordinator
    
    mock_stats_collector = Mock()
    mock_stats_collector.collect_stats.return_value = None
    coordinator_instance.stats_collector = mock_stats_collector
    
    stats = coordinator_instance._collect_statistics()
    
    assert stats is None


def test_check_alerts_with_manager(coordinator):
    """测试检查告警（有管理器）"""
    coordinator_instance, module = coordinator
    
    mock_alert_manager = Mock()
    mock_alert_manager.check_alerts.return_value = [{'alert_id': '1'}]
    coordinator_instance.alert_manager = mock_alert_manager
    
    stats = {'pool_size': 10}
    
    with patch.object(coordinator_instance, '_publish_alerts_detected_event', Mock()):
        alerts = coordinator_instance._check_alerts(stats)
        
        assert len(alerts) == 1
        assert alerts[0]['alert_id'] == '1'
        mock_alert_manager.check_alerts.assert_called_once_with(stats)


def test_check_alerts_no_manager(coordinator):
    """测试检查告警（无管理器）"""
    coordinator_instance, module = coordinator
    
    alerts = coordinator_instance._check_alerts({'test': 1})
    
    assert alerts == []


def test_check_alerts_no_stats(coordinator):
    """测试检查告警（无统计信息）"""
    coordinator_instance, module = coordinator
    
    mock_alert_manager = Mock()
    coordinator_instance.alert_manager = mock_alert_manager
    
    alerts = coordinator_instance._check_alerts(None)
    
    assert alerts == []
    mock_alert_manager.check_alerts.assert_not_called()


def test_export_metrics_success(coordinator):
    """测试导出指标（成功）"""
    coordinator_instance, module = coordinator
    
    mock_metrics_exporter = Mock()
    mock_metrics_exporter.export_metrics.return_value = True
    coordinator_instance.metrics_exporter = mock_metrics_exporter
    
    stats = {'pool_size': 10}
    
    with patch.object(coordinator_instance, '_publish_metrics_exported_event', Mock()):
        result = coordinator_instance._export_metrics(stats)
        
        assert result is True
        mock_metrics_exporter.export_metrics.assert_called_once_with(stats)


def test_export_metrics_no_exporter(coordinator):
    """测试导出指标（无导出器）"""
    coordinator_instance, module = coordinator
    
    result = coordinator_instance._export_metrics({'test': 1})
    
    assert result is False


def test_export_metrics_no_stats(coordinator):
    """测试导出指标（无统计信息）"""
    coordinator_instance, module = coordinator
    
    mock_metrics_exporter = Mock()
    coordinator_instance.metrics_exporter = mock_metrics_exporter
    
    result = coordinator_instance._export_metrics(None)
    
    assert result is False
    mock_metrics_exporter.export_metrics.assert_not_called()


def test_perform_monitoring_steps(coordinator):
    """测试执行监控步骤"""
    coordinator_instance, module = coordinator
    
    mock_stats_collector = Mock()
    mock_stats_collector.collect_stats.return_value = {'pool_size': 10}
    coordinator_instance.stats_collector = mock_stats_collector
    
    mock_alert_manager = Mock()
    mock_alert_manager.check_alerts.return_value = []
    coordinator_instance.alert_manager = mock_alert_manager
    
    mock_metrics_exporter = Mock()
    mock_metrics_exporter.export_metrics.return_value = True
    coordinator_instance.metrics_exporter = mock_metrics_exporter
    
    with patch.object(coordinator_instance, '_publish_stats_collected_event', Mock()):
        with patch.object(coordinator_instance, '_publish_metrics_exported_event', Mock()):
            result = coordinator_instance._perform_monitoring_steps()
            
            assert 'stats' in result
            assert 'alerts' in result
            assert 'export_success' in result
            assert result['stats'] == {'pool_size': 10}
            assert result['export_success'] is True


def test_execute_monitoring_cycle_success(coordinator):
    """测试执行监控周期（成功）"""
    coordinator_instance, module = coordinator
    
    mock_result = {'stats': {}, 'alerts': [], 'export_success': True}
    
    with patch.object(coordinator_instance, '_perform_monitoring_steps', return_value=mock_result):
        with patch.object(coordinator_instance, '_publish_cycle_started_event', Mock()):
            with patch.object(coordinator_instance, '_publish_cycle_completed_event', Mock()):
                coordinator_instance._execute_monitoring_cycle()
                
                coordinator_instance._publish_cycle_started_event.assert_called_once()
                coordinator_instance._publish_cycle_completed_event.assert_called_once_with(mock_result)


def test_execute_monitoring_cycle_exception(coordinator):
    """测试执行监控周期（异常）"""
    coordinator_instance, module = coordinator
    
    error = RuntimeError("Cycle error")
    
    with patch.object(coordinator_instance, '_perform_monitoring_steps', side_effect=error):
        with patch.object(coordinator_instance, '_publish_cycle_started_event', Mock()):
            with patch.object(coordinator_instance, '_publish_cycle_error_event', Mock()):
                coordinator_instance._execute_monitoring_cycle()
                
                coordinator_instance._publish_cycle_error_event.assert_called_once_with(error)


def test_monitoring_loop(coordinator):
    """测试监控循环"""
    coordinator_instance, module = coordinator
    
    coordinator_instance.monitoring_active = True
    coordinator_instance.config.collection_interval = 0.01  # 很短的间隔
    
    call_count = {'value': 0}
    
    def mock_execute():
        call_count['value'] += 1
        if call_count['value'] >= 3:
            coordinator_instance.monitoring_active = False
    
    with patch.object(coordinator_instance, '_execute_monitoring_cycle', side_effect=mock_execute):
        with patch.object(module.time, 'sleep', Mock()):
            coordinator_instance._monitoring_loop()
            
            assert call_count['value'] == 3


def test_monitoring_loop_exception(coordinator):
    """测试监控循环（异常）"""
    coordinator_instance, module = coordinator
    
    coordinator_instance.monitoring_active = True
    coordinator_instance.config.collection_interval = 0.01
    
    call_count = {'value': 0}
    
    def mock_execute():
        call_count['value'] += 1
        if call_count['value'] == 1:
            raise RuntimeError("Loop error")
        if call_count['value'] >= 2:
            coordinator_instance.monitoring_active = False
    
    with patch.object(coordinator_instance, '_execute_monitoring_cycle', side_effect=mock_execute):
        with patch.object(module.time, 'sleep', Mock()):
            coordinator_instance._monitoring_loop()
            
            # 应该继续执行，即使有异常
            assert call_count['value'] >= 2


def test_handle_control_event_start(coordinator):
    """测试处理控制事件（启动）"""
    coordinator_instance, module = coordinator
    
    mock_message = Mock()
    mock_message.topic = "monitoring.control.start"
    
    with patch.object(coordinator_instance, 'start_monitoring', return_value=True) as mock_start:
        coordinator_instance._handle_control_event(mock_message)
        mock_start.assert_called_once()


def test_handle_control_event_stop(coordinator):
    """测试处理控制事件（停止）"""
    coordinator_instance, module = coordinator
    
    mock_message = Mock()
    mock_message.topic = "monitoring.control.stop"
    
    with patch.object(coordinator_instance, 'stop_monitoring', return_value=True) as mock_stop:
        coordinator_instance._handle_control_event(mock_message)
        mock_stop.assert_called_once()


def test_handle_control_event_restart(coordinator):
    """测试处理控制事件（重启）"""
    coordinator_instance, module = coordinator
    
    mock_message = Mock()
    mock_message.topic = "monitoring.control.restart"
    
    with patch.object(coordinator_instance, 'stop_monitoring', return_value=True) as mock_stop:
        with patch.object(coordinator_instance, 'start_monitoring', return_value=True) as mock_start:
            with patch.object(module.time, 'sleep', Mock()):
                coordinator_instance._handle_control_event(mock_message)
                
                mock_stop.assert_called_once()
                mock_start.assert_called_once()


def test_handle_component_status_event_error(coordinator):
    """测试处理组件状态事件（错误）"""
    coordinator_instance, module = coordinator
    
    mock_message = Mock()
    mock_message.topic = "component.status.error"
    mock_message.payload = {
        'component': 'test_component',
        'status': 'error',
        'error': 'Test error'
    }
    
    # 使用capsys来捕获print输出
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        coordinator_instance._handle_component_status_event(mock_message)
        output = sys.stdout.getvalue()
        assert 'test_component' in output or 'Test error' in output
    finally:
        sys.stdout = old_stdout


def test_context_manager(coordinator):
    """测试上下文管理器"""
    coordinator_instance, module = coordinator
    
    mock_thread = Mock()
    mock_thread.is_alive.return_value = True
    mock_thread.start = Mock()
    mock_thread.join = Mock()
    
    with patch.object(module.threading, 'Thread', return_value=mock_thread):
        with coordinator_instance as coord:
            assert coord.monitoring_active is True
        
        # 退出上下文后应该停止监控
        assert coordinator_instance.monitoring_active is False


def test_publish_events(coordinator):
    """测试发布事件方法"""
    coordinator_instance, module = coordinator
    
    with patch.object(module, 'publish_event', Mock()) as mock_publish:
        # 测试发布统计收集事件
        coordinator_instance._publish_stats_collected_event({'test': 1})
        assert mock_publish.called
        
        # 测试发布告警检测事件
        mock_publish.reset_mock()
        coordinator_instance._publish_alerts_detected_event([{'alert': 1}])
        assert mock_publish.called
        
        # 测试发布指标导出事件
        mock_publish.reset_mock()
        coordinator_instance._publish_metrics_exported_event(True, {'test': 1})
        assert mock_publish.called
        
        # 测试发布周期完成事件
        mock_publish.reset_mock()
        coordinator_instance._publish_cycle_completed_event({
            'stats': {'test': 1},
            'alerts': [],
            'export_success': True
        })
        assert mock_publish.called
        
        # 测试发布周期错误事件
        mock_publish.reset_mock()
        coordinator_instance._publish_cycle_error_event(RuntimeError("Test error"))
        assert mock_publish.called


def test_publish_cycle_started_event(coordinator):
    """测试发布周期开始事件"""
    coordinator_instance, module = coordinator
    
    with patch.object(module, 'publish_event', Mock()) as mock_publish:
        coordinator_instance._publish_cycle_started_event()
        
        mock_publish.assert_called_once()
        call_args = mock_publish.call_args
        assert call_args[0][0] == "monitoring.cycle.started"
        assert 'pool_name' in call_args[0][1]
        assert 'timestamp' in call_args[0][1]

