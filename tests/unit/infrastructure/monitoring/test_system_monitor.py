import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from src.infrastructure.monitoring.system_monitor import SystemMonitor, ResourceMonitor
import os
import copy

# Fixtures
@pytest.fixture
def mock_psutil():
    """创建mock的psutil"""
    mock_psutil = MagicMock()
    
    # 模拟CPU信息
    mock_psutil.cpu_count.return_value = 8
    mock_psutil.cpu_percent.return_value = 45.5
    mock_psutil.boot_time.return_value = 1640995200  # 2022-01-01 00:00:00
    
    # 模拟内存信息
    mock_memory = MagicMock()
    mock_memory.total = 8589934592  # 8GB
    mock_memory.available = 4294967296  # 4GB
    mock_memory.used = 4294967296  # 4GB
    mock_memory.percent = 50.0
    mock_psutil.virtual_memory.return_value = mock_memory
    
    # 模拟磁盘信息
    mock_disk = MagicMock()
    mock_disk.total = 107374182400  # 100GB
    mock_disk.used = 53687091200  # 50GB
    mock_disk.free = 53687091200  # 50GB
    mock_disk.percent = 50.0
    mock_psutil.disk_usage.return_value = mock_disk
    
    # 模拟网络信息
    mock_net_io = MagicMock()
    mock_net_io.bytes_sent = 1024000
    mock_net_io.bytes_recv = 2048000
    mock_net_io.packets_sent = 1000
    mock_net_io.packets_recv = 2000
    mock_psutil.net_io_counters.return_value = mock_net_io
    
    # 模拟进程信息
    mock_psutil.pids.return_value = [1, 2, 3, 4, 5]
    
    return mock_psutil

@pytest.fixture
def mock_os():
    """创建mock的os"""
    mock_os = MagicMock()
    mock_os.getloadavg.return_value = (1.5, 1.2, 0.8)
    return mock_os

@pytest.fixture
def mock_socket():
    """创建mock的socket"""
    mock_socket = MagicMock()
    mock_socket.gethostname.return_value = "test-host"
    return mock_socket

@pytest.fixture
def system_monitor(mock_psutil, mock_os, mock_socket):
    """创建SystemMonitor实例"""
    return SystemMonitor(
        check_interval=1.0,
        psutil_mock=mock_psutil,
        os_mock=mock_os,
        socket_mock=mock_socket
    )

# 测试用例
class TestSystemMonitor:
    """SystemMonitor测试类"""

    def test_init_with_test_hooks(self, mock_psutil, mock_os, mock_socket):
        """测试使用测试钩子初始化"""
        monitor = SystemMonitor(
            check_interval=1.0,
            psutil_mock=mock_psutil,
            os_mock=mock_os,
            socket_mock=mock_socket
        )
        
        assert monitor.psutil == mock_psutil
        assert monitor.os == mock_os
        assert monitor.socket == mock_socket
        assert monitor.check_interval == 1.0

    def test_init_without_test_hooks(self):
        """测试不使用测试钩子初始化"""
        # 这个测试验证当不提供测试钩子时，会使用真实的模块
        monitor = SystemMonitor(check_interval=1.0)
        
        # 验证使用了真实的模块
        assert monitor.psutil is not None
        assert monitor.os is not None
        assert monitor.socket is not None
        assert monitor.check_interval == 1.0

    def test_get_system_info(self, system_monitor, mock_socket, mock_psutil):
        """测试获取系统信息"""
        info = system_monitor._get_system_info()
        
        assert info['hostname'] == 'test-host'
        assert info['cpu_count'] == 8
        assert 'boot_time' in info
        assert 'python_version' in info

    def test_collect_system_stats(self, system_monitor, mock_psutil, mock_os):
        """测试收集系统统计数据"""
        stats = system_monitor._collect_system_stats()
        
        assert 'timestamp' in stats
        assert 'cpu' in stats
        assert 'memory' in stats
        assert 'disk' in stats
        assert 'network' in stats
        assert 'process' in stats
        
        # 验证CPU数据
        assert stats['cpu']['percent'] == 45.5
        assert stats['cpu']['load_avg'] == [1.5, 1.2, 0.8]
        
        # 验证内存数据
        assert stats['memory']['total'] == 8589934592
        assert stats['memory']['percent'] == 50.0
        
        # 验证磁盘数据
        assert stats['disk']['total'] == 107374182400
        assert stats['disk']['percent'] == 50.0
        
        # 验证网络数据
        assert stats['network']['bytes_sent'] == 1024000
        assert stats['network']['bytes_recv'] == 2048000
        
        # 验证进程数据
        assert stats['process']['count'] == 5

    def test_get_load_avg_with_load(self, system_monitor, mock_os):
        """测试获取系统负载 - 有负载数据"""
        load_avg = system_monitor._get_load_avg()
        
        assert load_avg == [1.5, 1.2, 0.8]
        mock_os.getloadavg.assert_called_once()

    def test_get_load_avg_without_load(self, system_monitor):
        """测试获取系统负载 - 无负载数据"""
        # 模拟os没有getloadavg方法
        system_monitor.os = MagicMock()
        delattr(system_monitor.os, 'getloadavg')
        
        load_avg = system_monitor._get_load_avg()
        
        assert load_avg is None

    def test_get_load_avg_exception(self, system_monitor):
        """测试获取系统负载 - 异常处理"""
        # 模拟getloadavg抛出异常
        system_monitor.os.getloadavg.side_effect = Exception("Test error")
        
        load_avg = system_monitor._get_load_avg()
        
        assert load_avg is None

    def test_check_system_status_normal(self, system_monitor):
        """测试检查系统状态 - 正常状态"""
        stats = {
            'cpu': {'percent': 50.0},
            'memory': {'percent': 60.0},
            'disk': {'percent': 70.0}
        }
        
        with patch.object(system_monitor, '_trigger_alert') as mock_trigger:
            system_monitor._check_system_status(stats)
            
            # 应该没有触发告警
            mock_trigger.assert_not_called()

    def test_check_system_status_critical_cpu(self, system_monitor):
        """测试检查系统状态 - CPU告警"""
        stats = {
            'cpu': {'percent': 95.0},
            'memory': {'percent': 60.0},
            'disk': {'percent': 70.0}
        }
        
        with patch.object(system_monitor, '_trigger_alert') as mock_trigger:
            system_monitor._check_system_status(stats)
            
            # 应该触发CPU告警
            mock_trigger.assert_called_once()
            alert = mock_trigger.call_args[0][0]
            assert alert['type'] == 'cpu'
            assert alert['level'] == 'critical'
            assert alert['value'] == 95.0

    def test_check_system_status_critical_memory(self, system_monitor):
        """测试检查系统状态 - 内存告警"""
        stats = {
            'cpu': {'percent': 50.0},
            'memory': {'percent': 95.0},
            'disk': {'percent': 70.0}
        }
        
        with patch.object(system_monitor, '_trigger_alert') as mock_trigger:
            system_monitor._check_system_status(stats)
            
            # 应该触发内存告警
            mock_trigger.assert_called_once()
            alert = mock_trigger.call_args[0][0]
            assert alert['type'] == 'memory'
            assert alert['level'] == 'critical'

    def test_check_system_status_critical_disk(self, system_monitor):
        """测试检查系统状态 - 磁盘告警"""
        stats = {
            'cpu': {'percent': 50.0},
            'memory': {'percent': 60.0},
            'disk': {'percent': 95.0}
        }
        
        with patch.object(system_monitor, '_trigger_alert') as mock_trigger:
            system_monitor._check_system_status(stats)
            
            # 应该触发磁盘告警
            mock_trigger.assert_called_once()
            alert = mock_trigger.call_args[0][0]
            assert alert['type'] == 'disk'
            assert alert['level'] == 'critical'

    def test_trigger_alert(self, system_monitor):
        """测试触发告警"""
        alert = {
            'type': 'cpu',
            'level': 'critical',
            'message': 'High CPU usage: 95%',
            'value': 95.0,
            'threshold': 90
        }
        
        mock_handler = Mock()
        system_monitor.alert_handlers = [mock_handler]
        
        system_monitor._trigger_alert(alert)
        
        mock_handler.assert_called_once_with('system', alert)

    def test_trigger_alert_handler_exception(self, system_monitor):
        """测试触发告警 - 处理器异常"""
        alert = {
            'type': 'cpu',
            'level': 'critical',
            'message': 'High CPU usage: 95%'
        }
        
        mock_handler = Mock(side_effect=Exception("Handler error"))
        system_monitor.alert_handlers = [mock_handler]
        
        # 不应该抛出异常
        system_monitor._trigger_alert(alert)
        
        mock_handler.assert_called_once()

    def test_start_stop_monitoring(self, system_monitor):
        """测试启动和停止监控"""
        assert not system_monitor._monitoring
        
        system_monitor.start_monitoring()
        assert system_monitor._monitoring
        assert system_monitor._monitor_thread is not None
        
        system_monitor.stop_monitoring()
        assert not system_monitor._monitoring

    def test_get_stats_empty(self, system_monitor):
        """测试获取统计数据 - 空数据"""
        stats = system_monitor.get_stats()
        
        assert stats == []

    def test_get_stats_with_data(self, system_monitor):
        """测试获取统计数据 - 有数据"""
        # 添加一些测试数据
        test_stats = [
            {'timestamp': '2023-01-01T00:00:00', 'cpu': {'percent': 50.0}},
            {'timestamp': '2023-01-01T00:01:00', 'cpu': {'percent': 60.0}},
            {'timestamp': '2023-01-01T00:02:00', 'cpu': {'percent': 70.0}}
        ]
        system_monitor._stats = test_stats
        
        stats = system_monitor.get_stats()
        
        assert len(stats) == 3
        assert stats == test_stats

    def test_get_stats_with_time_filter(self, system_monitor):
        """测试获取统计数据 - 时间过滤"""
        test_stats = [
            {'timestamp': '2023-01-01T00:00:00', 'cpu': {'percent': 50.0}},
            {'timestamp': '2023-01-01T00:01:00', 'cpu': {'percent': 60.0}},
            {'timestamp': '2023-01-01T00:02:00', 'cpu': {'percent': 70.0}}
        ]
        system_monitor._stats = test_stats
        
        # 过滤时间范围
        stats = system_monitor.get_stats(
            start_time='2023-01-01T00:01:00',
            end_time='2023-01-01T00:01:00'
        )
        
        assert len(stats) == 1
        assert stats[0]['timestamp'] == '2023-01-01T00:01:00'

    def test_get_summary_empty(self, system_monitor):
        """测试获取摘要 - 空数据"""
        summary = system_monitor.get_summary()
        
        assert summary == {}

    def test_get_summary_with_data(self, system_monitor):
        """测试获取摘要 - 有数据"""
        test_stats = [
            {
                'timestamp': '2023-01-01T00:00:00',
                'cpu': {'percent': 50.0},
                'memory': {'percent': 60.0},
                'disk': {'percent': 70.0}
            },
            {
                'timestamp': '2023-01-01T00:01:00',
                'cpu': {'percent': 60.0},
                'memory': {'percent': 70.0},
                'disk': {'percent': 80.0}
            }
        ]
        system_monitor._stats = test_stats
        
        summary = system_monitor.get_summary()
        
        assert 'system_info' in summary
        assert 'cpu' in summary
        assert 'memory' in summary
        assert 'disk' in summary
        assert 'period' in summary
        
        # 验证CPU统计
        assert summary['cpu']['avg'] == 55.0
        assert summary['cpu']['max'] == 60.0
        
        # 验证内存统计
        assert summary['memory']['avg'] == 65.0
        assert summary['memory']['max'] == 70.0
        
        # 验证磁盘统计
        assert summary['disk']['avg'] == 75.0
        assert summary['disk']['max'] == 80.0

    def test_monitor_loop_exception(self, system_monitor):
        """测试监控循环异常处理"""
        # 模拟_collect_system_stats抛出异常
        with patch.object(system_monitor, '_collect_system_stats', side_effect=Exception("Test error")):
            system_monitor._monitoring = True
            
            # 应该捕获异常并继续运行
            system_monitor._monitor_loop()
            
            # 验证_monitoring被设置为False
            assert not system_monitor._monitoring

class TestResourceMonitor:
    """ResourceMonitor测试类"""

    def test_init_with_monitor(self, system_monitor):
        """测试使用监控器初始化"""
        monitor = ResourceMonitor(monitor=system_monitor)
        
        assert monitor.monitor == system_monitor

    def test_init_without_monitor(self):
        """测试不使用监控器初始化"""
        with patch('src.infrastructure.monitoring.system_monitor.SystemMonitor') as mock_system_monitor:
            mock_instance = Mock()
            mock_system_monitor.return_value = mock_instance
            
            monitor = ResourceMonitor()
            
            assert monitor.monitor == mock_instance
            mock_system_monitor.assert_called_once()

    def test_decorator_function(self, system_monitor):
        """测试装饰器功能"""
        monitor = ResourceMonitor(monitor=system_monitor)
        
        @monitor
        def test_function():
            return "test result"
        
        # 模拟_collect_system_stats方法
        with patch.object(system_monitor, '_collect_system_stats') as mock_collect:
            mock_collect.return_value = {
                'cpu': {'percent': 50.0},
                'memory': {'used': 1000}
            }
            
            result = test_function()
            
            assert result == "test result"
            # 验证监控方法被调用
            assert mock_collect.call_count == 2  # 开始和结束时各一次

    def test_decorator_function_exception(self, system_monitor):
        """测试装饰器异常处理"""
        monitor = ResourceMonitor(monitor=system_monitor)
        
        @monitor
        def test_function():
            raise Exception("Test error")
        
        # 模拟_collect_system_stats方法
        with patch.object(system_monitor, '_collect_system_stats') as mock_collect:
            mock_collect.return_value = {
                'cpu': {'percent': 50.0},
                'memory': {'used': 1000}
            }
            
            # 应该抛出异常
            with pytest.raises(Exception, match="Test error"):
                test_function()
            
            # 验证监控方法仍然被调用
            assert mock_collect.call_count == 2  # 开始和结束时各一次
