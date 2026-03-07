"""
测试系统资源分析器

覆盖 system_resource_analyzer.py 中的所有类和功能
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.infrastructure.resource.core.system_resource_analyzer import SystemResourceAnalyzer


class TestSystemResourceAnalyzer:
    """SystemResourceAnalyzer 类测试"""

    def test_initialization(self):
        """测试初始化"""
        analyzer = SystemResourceAnalyzer()

        assert analyzer.logger is not None
        assert analyzer.error_handler is not None

    def test_initialization_with_components(self):
        """测试带组件初始化"""
        mock_logger = Mock()
        mock_error_handler = Mock()

        analyzer = SystemResourceAnalyzer(
            logger=mock_logger,
            error_handler=mock_error_handler
        )

        assert analyzer.logger == mock_logger
        assert analyzer.error_handler == mock_error_handler

    @patch('src.infrastructure.resource.core.system_resource_analyzer.psutil')
    def test_get_system_resources_basic(self, mock_psutil):
        """测试获取系统资源（基本深度）"""
        # Mock psutil 返回值
        mock_psutil.cpu_percent.return_value = 45.5
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.cpu_count.return_value = 8  # logical=True
        mock_psutil.virtual_memory.return_value.percent = 67.8
        mock_psutil.virtual_memory.return_value.used = 4 * 1024**3
        mock_psutil.virtual_memory.return_value.total = 8 * 1024**3

        analyzer = SystemResourceAnalyzer()

        resources = analyzer.get_system_resources("basic")

        assert isinstance(resources, dict)
        assert 'timestamp' in resources
        assert 'cpu' in resources
        assert 'memory' in resources
        assert 'threads' in resources
        assert 'io' in resources

    @patch('src.infrastructure.resource.core.system_resource_analyzer.psutil')
    def test_get_system_resources_detailed(self, mock_psutil):
        """测试获取系统资源（详细深度）"""
        # Mock psutil 返回值
        mock_psutil.cpu_percent.return_value = 45.5
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.cpu_count.return_value = 8  # logical=True
        mock_psutil.cpu_freq.return_value = [Mock(current=2500, min=800, max=3500)]
        mock_psutil.virtual_memory.return_value.percent = 67.8
        mock_psutil.virtual_memory.return_value.used = 4 * 1024**3
        mock_psutil.virtual_memory.return_value.total = 8 * 1024**3
        mock_psutil.swap_memory.return_value.percent = 12.3
        mock_psutil.net_io_counters.return_value = Mock(bytes_sent=1000, bytes_recv=2000)
        mock_psutil.disk_io_counters.return_value = Mock(read_bytes=5000, write_bytes=3000)

        analyzer = SystemResourceAnalyzer()

        resources = analyzer.get_system_resources("detailed")

        assert isinstance(resources, dict)
        assert 'timestamp' in resources
        assert 'cpu' in resources
        assert 'memory' in resources
        assert 'threads' in resources
        assert 'io' in resources

        # 检查详细模式下的额外字段
        assert 'frequency' in resources['cpu']
        assert 'swap_percent' in resources['memory']
        assert 'network' in resources['io']

    def test_get_system_resources_error_handling(self):
        """测试获取系统资源时的错误处理"""
        analyzer = SystemResourceAnalyzer()

        # Mock 让_get_cpu_resources抛出异常
        with patch.object(analyzer, '_get_cpu_resources', side_effect=Exception("Hardware error")):
            resources = analyzer.get_system_resources()

            # 应该返回错误响应
            assert 'error' in resources
            assert 'timestamp' in resources
            assert resources['error'] == "Hardware error"

    @patch('src.infrastructure.resource.core.system_resource_analyzer.psutil')
    def test_get_cpu_resources_basic(self, mock_psutil):
        """测试获取CPU资源（基本）"""
        mock_psutil.cpu_percent.return_value = 45.5
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.cpu_count.return_value = 8  # logical=True

        analyzer = SystemResourceAnalyzer()

        cpu_info = analyzer._get_cpu_resources("basic")

        assert isinstance(cpu_info, dict)
        assert 'usage_percent' in cpu_info
        assert 'count' in cpu_info
        assert 'count_logical' in cpu_info

    @patch('src.infrastructure.resource.core.system_resource_analyzer.psutil')
    def test_get_cpu_resources_detailed(self, mock_psutil):
        """测试获取CPU资源（详细）"""
        mock_psutil.cpu_percent.return_value = 45.5
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.cpu_count.return_value = 8  # logical=True
        mock_psutil.cpu_freq.return_value = [Mock(current=2500, min=800, max=3500)]

        analyzer = SystemResourceAnalyzer()

        cpu_info = analyzer._get_cpu_resources("detailed")

        assert isinstance(cpu_info, dict)
        assert 'usage_percent' in cpu_info
        assert 'count' in cpu_info
        assert 'count_logical' in cpu_info
        assert 'frequency' in cpu_info

    @patch('src.infrastructure.resource.core.system_resource_analyzer.psutil')
    def test_get_memory_resources_basic(self, mock_psutil):
        """测试获取内存资源（基本）"""
        mock_psutil.virtual_memory.return_value.percent = 67.8
        mock_psutil.virtual_memory.return_value.used = 4 * 1024**3
        mock_psutil.virtual_memory.return_value.total = 8 * 1024**3

        analyzer = SystemResourceAnalyzer()

        memory_info = analyzer._get_memory_resources("basic")

        assert isinstance(memory_info, dict)
        assert 'usage_percent' in memory_info
        assert 'used_gb' in memory_info
        assert 'total_gb' in memory_info

    @patch('src.infrastructure.resource.core.system_resource_analyzer.psutil')
    def test_get_memory_resources_detailed(self, mock_psutil):
        """测试获取内存资源（详细）"""
        mock_psutil.virtual_memory.return_value.percent = 67.8
        mock_psutil.virtual_memory.return_value.used = 4 * 1024**3
        mock_psutil.virtual_memory.return_value.total = 8 * 1024**3
        mock_psutil.swap_memory.return_value.percent = 12.3
        mock_psutil.swap_memory.return_value.used = 1 * 1024**3
        mock_psutil.swap_memory.return_value.total = 2 * 1024**3

        analyzer = SystemResourceAnalyzer()

        memory_info = analyzer._get_memory_resources("detailed")

        assert isinstance(memory_info, dict)
        assert 'usage_percent' in memory_info
        assert 'used_gb' in memory_info
        assert 'total_gb' in memory_info
        assert 'swap_percent' in memory_info
        assert 'swap_used_gb' in memory_info
        assert 'swap_total_gb' in memory_info

    @patch('src.infrastructure.resource.core.system_resource_analyzer.psutil')
    def test_get_thread_resources(self, mock_psutil):
        """测试获取线程资源"""
        mock_psutil.process_iter.return_value = [Mock(pid=1, num_threads=4), Mock(pid=2, num_threads=2)]

        analyzer = SystemResourceAnalyzer()

        thread_info = analyzer._get_thread_resources("basic")

        assert isinstance(thread_info, dict)
        assert 'process_thread_count' in thread_info
        assert 'system_thread_count' in thread_info

    @patch('src.infrastructure.resource.core.system_resource_analyzer.psutil')
    def test_get_io_resources_basic(self, mock_psutil):
        """测试获取I/O资源（基本）"""
        mock_psutil.disk_io_counters.return_value = Mock(read_bytes=5000, write_bytes=3000)

        analyzer = SystemResourceAnalyzer()

        io_info = analyzer._get_io_resources("basic")

        assert isinstance(io_info, dict)
        assert 'disk' in io_info
        assert 'network' in io_info
        assert 'read_bytes' in io_info['disk']
        assert 'write_bytes' in io_info['disk']

    @patch('src.infrastructure.resource.core.system_resource_analyzer.psutil')
    def test_get_io_resources_detailed(self, mock_psutil):
        """测试获取I/O资源（详细）"""
        mock_psutil.disk_io_counters.return_value = Mock(read_bytes=5000, write_bytes=3000)
        mock_psutil.net_io_counters.return_value = Mock(bytes_sent=1000, bytes_recv=2000)

        analyzer = SystemResourceAnalyzer()

        io_info = analyzer._get_io_resources("detailed")

        assert isinstance(io_info, dict)
        assert 'disk' in io_info
        assert 'network' in io_info
        assert 'disk_partitions' in io_info
        assert 'read_bytes' in io_info['disk']
        assert 'bytes_sent' in io_info['network']

    def test_get_resource_summary(self):
        """测试获取资源汇总"""
        analyzer = SystemResourceAnalyzer()

        summary = analyzer.get_resource_summary()

        assert isinstance(summary, dict)
        assert 'timestamp' in summary
        assert 'cpu_usage' in summary
        assert 'memory_usage' in summary
        assert 'thread_count' in summary

    def test_analyzer_initialization_with_custom_components(self):
        """测试分析器使用自定义组件初始化"""
        custom_logger = Mock()
        custom_error_handler = Mock()

        analyzer = SystemResourceAnalyzer(
            logger=custom_logger,
            error_handler=custom_error_handler
        )

        assert analyzer.logger == custom_logger
        assert analyzer.error_handler == custom_error_handler

    def test_system_resources_analysis_depths(self):
        """测试不同分析深度的系统资源获取"""
        analyzer = SystemResourceAnalyzer()

        # 测试所有支持的深度级别
        for depth in ["basic", "detailed", "comprehensive"]:
            resources = analyzer.get_system_resources(depth)

            assert isinstance(resources, dict)
            assert 'timestamp' in resources
            assert 'cpu' in resources
            assert 'memory' in resources
            assert 'threads' in resources
            assert 'io' in resources

    def test_io_resources_with_none_counters(self):
        """测试I/O资源获取当计数器为None时的情况"""
        analyzer = SystemResourceAnalyzer()

        # Mock psutil返回None
        with patch('psutil.disk_io_counters', return_value=None):
            with patch('psutil.net_io_counters', return_value=None):
                io_info = analyzer._get_io_resources("basic")

                assert isinstance(io_info, dict)
                assert 'disk' in io_info
                assert 'network' in io_info

                # 所有值应该为0
                assert io_info['disk']['read_count'] == 0
                assert io_info['disk']['write_count'] == 0
                assert io_info['disk']['read_bytes'] == 0
                assert io_info['disk']['write_bytes'] == 0

                assert io_info['network']['bytes_sent'] == 0
                assert io_info['network']['bytes_recv'] == 0
                assert io_info['network']['packets_sent'] == 0
                assert io_info['network']['packets_recv'] == 0

    def test_io_resources_detailed_with_partitions(self):
        """测试详细I/O资源获取包含分区信息"""
        analyzer = SystemResourceAnalyzer()

        # Mock分区信息
        mock_partition = Mock()
        mock_partition.device = '/dev/sda1'
        mock_partition.mountpoint = '/'

        mock_usage = Mock()
        mock_usage.percent = 75.5
        mock_usage.used = 100 * 1024**3  # 100GB
        mock_usage.total = 500 * 1024**3  # 500GB

        with patch('psutil.disk_io_counters') as mock_disk:
            with patch('psutil.net_io_counters') as mock_net:
                with patch('psutil.disk_partitions', return_value=[mock_partition]):
                    with patch('psutil.disk_usage', return_value=mock_usage):
                        io_info = analyzer._get_io_resources("detailed")

                        assert isinstance(io_info, dict)
                        assert 'disk' in io_info
                        assert 'network' in io_info
                        assert 'disk_partitions' in io_info

                        assert isinstance(io_info['disk_partitions'], list)
                        assert len(io_info['disk_partitions']) > 0

                        partition = io_info['disk_partitions'][0]
                        assert 'device' in partition
                        assert 'mountpoint' in partition
                        assert 'usage_percent' in partition
                        assert 'used_gb' in partition
                        assert 'total_gb' in partition

    def test_thread_resources_error_handling(self):
        """测试线程资源获取的错误处理"""
        analyzer = SystemResourceAnalyzer()

        # Mock psutil.Process 抛出异常
        with patch('psutil.Process', side_effect=Exception("Process access denied")):
            thread_info = analyzer._get_thread_resources("basic")

            assert isinstance(thread_info, dict)
            assert 'error' in thread_info
            assert 'Process access denied' in thread_info['error']

    def test_get_system_resources_with_invalid_depth(self):
        """测试使用无效深度参数获取系统资源"""
        analyzer = SystemResourceAnalyzer()

        # 无效的深度参数应该使用默认值
        resources = analyzer.get_system_resources("invalid_depth")

        assert isinstance(resources, dict)
        assert 'timestamp' in resources
        assert 'cpu' in resources
        assert 'memory' in resources
        # 应该按照默认深度("basic")处理
