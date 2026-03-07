"""
测试目标：提升resource/monitoring/health/health_metrics_collector.py的真实覆盖率
实际导入和使用src.infrastructure.resource.monitoring.health.health_metrics_collector模块
"""

from unittest.mock import Mock, patch
import pytest
import time

from src.infrastructure.resource.monitoring.health.health_metrics_collector import HealthMetricsCollector
from src.infrastructure.resource.models.alert_dataclasses import PerformanceMetrics


class TestHealthMetricsCollector:
    """测试HealthMetricsCollector类"""

    @pytest.fixture
    def mock_logger(self):
        """模拟logger"""
        return Mock()

    @pytest.fixture
    def collector(self, mock_logger):
        """创建收集器实例"""
        return HealthMetricsCollector(logger=mock_logger)

    @pytest.fixture
    def collector_with_config(self, mock_logger):
        """创建带有配置的收集器实例"""
        config = {'metrics_cache_ttl': 30}
        return HealthMetricsCollector(config=config, logger=mock_logger)

    def test_initialization_default_config(self, collector, mock_logger):
        """测试使用默认配置的初始化"""
        assert collector.config == {}
        assert collector.logger == mock_logger
        assert collector._metrics_cache is None
        assert collector._cache_timestamp == 0
        assert collector._cache_ttl == 60

    def test_initialization_custom_config(self, collector_with_config, mock_logger):
        """测试使用自定义配置的初始化"""
        assert collector_with_config.config == {'metrics_cache_ttl': 30}
        assert collector_with_config.logger == mock_logger
        assert collector_with_config._cache_ttl == 30

    def test_initialization_without_logger(self):
        """测试不提供logger时的初始化"""
        collector = HealthMetricsCollector()

        assert collector.logger is not None
        assert hasattr(collector.logger, 'log_error')

    @patch('src.infrastructure.resource.monitoring.health.health_metrics_collector.HealthMetricsCollector._collect_thread_metrics')
    @patch('src.infrastructure.resource.monitoring.health.health_metrics_collector.HealthMetricsCollector._collect_process_metrics')
    @patch('src.infrastructure.resource.monitoring.health.health_metrics_collector.HealthMetricsCollector._collect_cpu_metrics')
    @patch('src.infrastructure.resource.monitoring.health.health_metrics_collector.HealthMetricsCollector._collect_memory_metrics')
    @patch('src.infrastructure.resource.monitoring.health.health_metrics_collector.HealthMetricsCollector._collect_disk_metrics')
    @patch('src.infrastructure.resource.monitoring.health.health_metrics_collector.HealthMetricsCollector._collect_network_metrics')
    def test_collect_current_metrics_success(self, mock_network, mock_disk, mock_memory, mock_cpu, mock_process, mock_thread, collector):
        """测试成功收集当前指标"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = 60.0
        mock_disk.return_value = 40.0
        mock_network.return_value = {'bytes_sent': 1000, 'bytes_recv': 2000}
        mock_process.return_value = 5
        mock_thread.return_value = 8

        with patch('time.time', return_value=1000.0):
            result = collector.collect_current_metrics()

            assert isinstance(result, PerformanceMetrics)
            assert result.cpu_usage == 50.0
            assert result.memory_usage == 60.0
            assert result.disk_usage == 40.0
            assert result.network_latency == 0.0
            assert result.test_execution_time == 0.0
            assert result.test_success_rate == 1.0
            assert result.active_threads == 8

            # 验证缓存
            assert collector._metrics_cache == result
            assert collector._cache_timestamp == 1000.0

    @patch('src.infrastructure.resource.monitoring.health.health_metrics_collector.HealthMetricsCollector._collect_cpu_metrics')
    def test_collect_current_metrics_failure(self, mock_cpu, collector, mock_logger):
        """测试收集指标失败"""
        mock_cpu.side_effect = Exception("Collection failed")

        result = collector.collect_current_metrics()

        assert result is None
        mock_logger.log_error.assert_called_once()

    def test_collect_current_metrics_from_cache(self, collector):
        """测试从缓存收集指标"""
        # 设置缓存
        cached_metrics = PerformanceMetrics(
            cpu_usage=45.0,
            memory_usage=55.0,
            disk_usage=35.0,
            network_usage=25.0,
            timestamp=1000.0
        )
        collector._metrics_cache = cached_metrics
        collector._cache_timestamp = 1000.0

        # 模拟缓存仍然有效
        with patch.object(collector, '_is_cache_valid', return_value=True):
            result = collector.collect_current_metrics()

            assert result == cached_metrics

    @patch('psutil.cpu_percent')
    def test_collect_cpu_metrics_success(self, mock_cpu_percent, collector):
        """测试成功收集CPU指标"""
        mock_cpu_percent.return_value = 75.5

        result = collector._collect_cpu_metrics()

        assert result == 75.5
        mock_cpu_percent.assert_called_once_with(interval=1)

    @patch('psutil.cpu_percent')
    def test_collect_cpu_metrics_failure(self, mock_cpu_percent, collector, mock_logger):
        """测试收集CPU指标失败"""
        mock_cpu_percent.side_effect = Exception("CPU error")

        result = collector._collect_cpu_metrics()

        assert result == 0.0
        mock_logger.log_error.assert_called_once()

    @patch('psutil.virtual_memory')
    def test_collect_memory_metrics_success(self, mock_virtual_memory, collector):
        """测试成功收集内存指标"""
        mock_memory = Mock()
        mock_memory.percent = 85.3
        mock_virtual_memory.return_value = mock_memory

        result = collector._collect_memory_metrics()

        assert result == 85.3
        mock_virtual_memory.assert_called_once()

    @patch('psutil.virtual_memory')
    def test_collect_memory_metrics_failure(self, mock_virtual_memory, collector, mock_logger):
        """测试收集内存指标失败"""
        mock_virtual_memory.side_effect = Exception("Memory error")

        result = collector._collect_memory_metrics()

        assert result == 0.0
        mock_logger.log_error.assert_called_once()

    @patch('psutil.disk_usage')
    def test_collect_disk_metrics_success(self, mock_disk_usage, collector):
        """测试成功收集磁盘指标"""
        mock_disk = Mock()
        mock_disk.percent = 72.1
        mock_disk_usage.return_value = mock_disk

        result = collector._collect_disk_metrics()

        assert result == 72.1
        mock_disk_usage.assert_called_once_with('/')

    @patch('psutil.disk_usage')
    def test_collect_disk_metrics_failure(self, mock_disk_usage, collector, mock_logger):
        """测试收集磁盘指标失败"""
        mock_disk_usage.side_effect = Exception("Disk error")

        result = collector._collect_disk_metrics()

        assert result == 0.0
        mock_logger.log_error.assert_called_once()

    @patch('psutil.net_io_counters')
    def test_collect_network_metrics_success(self, mock_net_io, collector):
        """测试成功收集网络指标"""
        mock_net = Mock()
        mock_net.bytes_sent = 1500
        mock_net.bytes_recv = 2500
        mock_net.packets_sent = 10
        mock_net.packets_recv = 15
        mock_net.errin = 0
        mock_net.errout = 0
        mock_net.dropin = 0
        mock_net.dropout = 0
        mock_net_io.return_value = mock_net

        result = collector._collect_network_metrics()

        expected = {
            'bytes_sent': 1500,
            'bytes_recv': 2500,
            'packets_sent': 10,
            'packets_recv': 15,
            'errors_in': 0,
            'errors_out': 0,
            'drops_in': 0,
            'drops_out': 0
        }
        assert result == expected
        mock_net_io.assert_called_once_with(pernic=False)

    @patch('psutil.net_io_counters')
    def test_collect_network_metrics_failure(self, mock_net_io, collector, mock_logger):
        """测试收集网络指标失败"""
        mock_net_io.side_effect = Exception("Network error")

        result = collector._collect_network_metrics()

        expected = {
            'bytes_sent': 0,
            'bytes_recv': 0,
            'packets_sent': 0,
            'packets_recv': 0,
            'errors_in': 0,
            'errors_out': 0,
            'drops_in': 0,
            'drops_out': 0
        }
        assert result == expected
        mock_logger.log_error.assert_called_once()

    def test_is_cache_valid_expired(self, collector):
        """测试缓存过期"""
        collector._cache_timestamp = time.time() - 120  # 2分钟前
        collector._cache_ttl = 60  # 60秒TTL

        result = collector._is_cache_valid()

        assert result is False

    def test_is_cache_valid_valid(self, collector):
        """测试缓存有效"""
        collector._cache_timestamp = time.time() - 30  # 30秒前
        collector._cache_ttl = 60  # 60秒TTL

        result = collector._is_cache_valid()

        assert result is True

    def test_is_cache_valid_no_cache(self, collector):
        """测试无缓存"""
        collector._metrics_cache = None

        result = collector._is_cache_valid()

        assert result is False

    def test_clear_cache(self, collector):
        """测试清除缓存"""
        collector._metrics_cache = Mock()
        collector._cache_timestamp = 1000.0

        collector.clear_cache()

        assert collector._metrics_cache is None
        assert collector._cache_timestamp == 0

    def test_get_cache_info(self, collector):
        """测试获取缓存信息"""
        collector._cache_timestamp = 1000.0
        collector._cache_ttl = 60

        info = collector.get_cache_info()

        assert info['cache_timestamp'] == 1000.0
        assert info['cache_ttl'] == 60
        assert 'cache_age' in info
        assert info['has_cache'] is False  # 因为_metrics_cache为None

    def test_get_cache_info_with_cache(self, collector):
        """测试获取有缓存时的缓存信息"""
        collector._metrics_cache = Mock()
        collector._cache_timestamp = time.time() - 30

        info = collector.get_cache_info()

        assert info['has_cache'] is True
        assert 'cache_age' in info
        assert info['cache_age'] > 0

    def test_get_collector_status(self, collector):
        """测试获取收集器状态"""
        status = collector.get_collector_status()

        assert 'cache_enabled' in status
        assert 'cache_ttl' in status
        assert 'last_collection' in status
        assert status['cache_enabled'] is True
        assert status['cache_ttl'] == 60
