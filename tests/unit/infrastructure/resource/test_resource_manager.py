import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.resource.resource_manager import ResourceManager, ResourceAllocationError

class TestResourceManager:
    """ResourceManager类测试"""

    @pytest.fixture
    def resource_manager(self):
        """创建资源管理器实例"""
        return ResourceManager(
            cpu_threshold=80.0,
            mem_threshold=85.0,
            disk_threshold=90.0,
            check_interval=1.0
        )

    def test_resource_manager_init(self, resource_manager):
        """测试资源管理器初始化"""
        assert resource_manager.cpu_threshold == 80.0
        assert resource_manager.mem_threshold == 85.0
        assert resource_manager.disk_threshold == 90.0
        assert resource_manager.check_interval == 1.0
        assert resource_manager._monitoring is False
        assert resource_manager._stats == []

    def test_start_monitoring(self, resource_manager):
        """测试启动监控"""
        resource_manager.start_monitoring()
        
        assert resource_manager._monitoring is True
        assert resource_manager._monitor_thread is not None
        assert resource_manager._monitor_thread.is_alive()
        
        resource_manager.stop_monitoring()

    def test_stop_monitoring(self, resource_manager):
        """测试停止监控"""
        resource_manager.start_monitoring()
        resource_manager.stop_monitoring()
        
        assert resource_manager._monitoring is False

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_get_current_stats(self, mock_net_io, mock_disk, mock_mem, mock_cpu, resource_manager):
        """测试获取当前统计信息"""
        # Mock系统调用
        mock_cpu.return_value = 45.5
        mock_mem.return_value = Mock(
            total=8589934592,  # 8GB
            available=4294967296,  # 4GB
            used=4294967296,  # 4GB
            percent=50.0
        )
        mock_disk.return_value = Mock(
            total=1000000000000,  # 1TB
            used=500000000000,  # 500GB
            free=500000000000,  # 500GB
            percent=50.0
        )
        mock_net_io.return_value = Mock(
            bytes_sent=1000000,
            bytes_recv=2000000
        )
        
        stats = resource_manager.get_current_stats()
        
        assert 'timestamp' in stats
        assert 'cpu' in stats
        assert 'memory' in stats
        assert 'disk' in stats
        assert 'network' in stats
        assert 'gpu' in stats
        
        assert stats['cpu']['percent'] == 45.5
        assert stats['memory']['percent'] == 50.0
        assert stats['disk']['percent'] == 50.0

    def test_get_stats_with_time_range(self, resource_manager):
        """测试获取指定时间范围的统计信息"""
        # 添加一些模拟数据
        resource_manager._stats = [
            {'timestamp': '2024-01-01T10:00:00', 'cpu': {'percent': 50}},
            {'timestamp': '2024-01-01T10:05:00', 'cpu': {'percent': 60}},
            {'timestamp': '2024-01-01T10:10:00', 'cpu': {'percent': 70}}
        ]
        
        stats = resource_manager.get_stats(
            start_time='2024-01-01T10:05:00',
            end_time='2024-01-01T10:10:00'
        )
        
        assert len(stats) == 2
        assert stats[0]['timestamp'] == '2024-01-01T10:05:00'
        assert stats[1]['timestamp'] == '2024-01-01T10:10:00'

    def test_get_summary(self, resource_manager):
        """测试获取统计摘要"""
        # 添加模拟数据
        resource_manager._stats = [
            {'cpu': {'percent': 50}, 'memory': {'percent': 60}},
            {'cpu': {'percent': 60}, 'memory': {'percent': 70}},
            {'cpu': {'percent': 70}, 'memory': {'percent': 80}}
        ]
        
        summary = resource_manager.get_summary()
        
        assert 'cpu_avg' in summary
        assert 'memory_avg' in summary
        assert 'disk_avg' in summary
        assert 'total_samples' in summary

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_threshold_checking(self, mock_disk, mock_mem, mock_cpu, resource_manager):
        """测试阈值检查"""
        # Mock高CPU使用率
        mock_cpu.return_value = 85.0  # 超过阈值
        mock_mem.return_value = Mock(
            total=8589934592,
            available=4294967296,
            used=4294967296,
            percent=50.0  # 未超过阈值
        )
        mock_disk.return_value = Mock(
            total=1000000000000,
            used=500000000000,
            free=500000000000,
            percent=50.0  # 未超过阈值
        )
        
        # 模拟告警处理器
        alert_handler = Mock()
        resource_manager.alert_handlers = [alert_handler]
        
        stats = resource_manager.get_current_stats()
        resource_manager._check_thresholds(stats)
        
        # 检查是否触发了CPU告警
        alert_handler.assert_called_once()
        alert_call = alert_handler.call_args[0][0]
        assert alert_call['type'] == 'cpu'
        assert alert_call['level'] == 'warning'

    def test_gpu_manager_initialization(self, resource_manager):
        """测试GPU管理器初始化"""
        # 测试无PyTorch环境
        resource_manager_no_torch = ResourceManager()
        assert resource_manager_no_torch.gpu_manager is None

    def test_resource_allocation_error(self):
        """测试资源分配异常"""
        with pytest.raises(ResourceAllocationError):
            raise ResourceAllocationError("Resource allocation failed")

    def test_monitor_thread_safety(self, resource_manager):
        """测试监控线程安全性"""
        resource_manager.start_monitoring()
        
        # 在另一个线程中访问统计数据
        def access_stats():
            for _ in range(10):
                with resource_manager._lock:
                    _ = len(resource_manager._stats)
                time.sleep(0.01)
        
        thread = threading.Thread(target=access_stats)
        thread.start()
        thread.join()
        
        resource_manager.stop_monitoring()

    def test_stats_history_limit(self, resource_manager):
        """测试统计历史数据限制"""
        # 添加超过限制的数据
        for i in range(1100):
            resource_manager._stats.append({'timestamp': f'2024-01-01T{i:02d}:00:00'})
        
        # 检查是否限制在1000条
        assert len(resource_manager._stats) == 1000

    def test_alert_handlers(self, resource_manager):
        """测试告警处理器"""
        alert_handler1 = Mock()
        alert_handler2 = Mock()
        
        resource_manager.alert_handlers = [alert_handler1, alert_handler2]
        
        # 模拟告警
        alert = {
            'type': 'cpu',
            'level': 'warning',
            'message': 'Test alert',
            'value': 85.0,
            'threshold': 80.0
        }
        
        resource_manager._trigger_alert(alert)
        
        # 检查两个处理器都被调用
        alert_handler1.assert_called_once_with(alert)
        alert_handler2.assert_called_once_with(alert)

    def test_resource_manager_integration(self, resource_manager):
        """测试资源管理器集成功能"""
        # 启动监控
        resource_manager.start_monitoring()
        
        # 等待一段时间收集数据
        time.sleep(2)
        
        # 检查是否有统计数据
        assert len(resource_manager._stats) > 0
        
        # 获取摘要
        summary = resource_manager.get_summary()
        assert summary['total_samples'] > 0
        
        # 停止监控
        resource_manager.stop_monitoring()

    def test_error_handling_in_monitor_loop(self, resource_manager):
        """测试监控循环中的错误处理"""
        # 模拟psutil调用失败
        with patch('psutil.cpu_percent', side_effect=Exception("CPU error")):
            resource_manager.start_monitoring()
            time.sleep(1)
            resource_manager.stop_monitoring()
            
            # 应该不会因为异常而崩溃

    def test_gpu_stats_handling(self, resource_manager):
        """测试GPU统计信息处理"""
        # 模拟GPU管理器
        mock_gpu_manager = Mock()
        mock_gpu_manager.get_gpu_stats.return_value = [
            {'name': 'GPU 0', 'memory_used': 2048, 'memory_total': 8192}
        ]
        resource_manager.gpu_manager = mock_gpu_manager
        
        stats = resource_manager.get_current_stats()
        
        assert stats['gpu'] is not None
        assert len(stats['gpu']) == 1
        assert stats['gpu'][0]['name'] == 'GPU 0'

    def test_gpu_stats_error_handling(self, resource_manager):
        """测试GPU统计信息错误处理"""
        # 模拟GPU管理器抛出异常
        mock_gpu_manager = Mock()
        mock_gpu_manager.get_gpu_stats.side_effect = Exception("GPU error")
        resource_manager.gpu_manager = mock_gpu_manager
        
        stats = resource_manager.get_current_stats()
        
        # 应该返回None而不是崩溃
        assert stats['gpu'] is None

    def test_load_average_handling(self, resource_manager):
        """测试系统负载处理"""
        # 在Windows系统上测试
        with patch('platform.system', return_value='Windows'):
            stats = resource_manager.get_current_stats()
            assert stats['cpu']['load_avg'] is None

    def test_resource_manager_performance(self, resource_manager):
        """测试资源管理器性能"""
        start_time = time.time()
        
        # 快速获取多次统计信息
        for _ in range(100):
            resource_manager.get_current_stats()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 应该在合理时间内完成
        assert execution_time < 10.0  # 10秒内完成100次调用

def test_resource_allocation_error_message():
    """测试资源分配异常消息"""
    error = ResourceAllocationError("Custom error message")
    assert str(error) == "Custom error message"

def test_resource_manager_with_custom_alert_handlers():
    """测试带自定义告警处理器的资源管理器"""
    alert_handler = Mock()
    
    resource_manager = ResourceManager(
        alert_handlers=[alert_handler]
    )
    
    # 模拟告警
    alert = {
        'type': 'memory',
        'level': 'critical',
        'message': 'Memory critical',
        'value': 95.0,
        'threshold': 90.0
    }
    
    resource_manager._trigger_alert(alert)
    alert_handler.assert_called_once_with(alert)

def test_resource_manager_thread_safety_stress():
    """测试资源管理器线程安全压力测试"""
    resource_manager = ResourceManager()
    
    def stress_worker():
        for _ in range(50):
            resource_manager.get_current_stats()
            time.sleep(0.01)
    
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=stress_worker)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # 应该没有异常发生
