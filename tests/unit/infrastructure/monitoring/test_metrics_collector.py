#!/usr/bin/env python3
"""
RQA2025 基础设施层指标收集器单元测试

测试 MetricsCollector 的功能和性能。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.infrastructure.monitoring.services.metrics_collector import MetricsCollector


class TestMetricsCollector(unittest.TestCase):
    """指标收集器测试类"""

    def setUp(self):
        """测试前准备"""
        self.collector = MetricsCollector()

    def tearDown(self):
        """测试后清理"""
        self.collector.clear_cache()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.collector.collection_stats, dict)
        self.assertIn('total_collections', self.collector.collection_stats)
        self.assertEqual(self.collector.collection_stats['total_collections'], 0)
        self.assertEqual(self.collector._cache_timeout, 30)

    def test_collect_all_metrics_structure(self):
        """测试收集所有指标的数据结构"""
        metrics = self.collector.collect_all_metrics()

        # 检查基本结构
        required_keys = [
            'timestamp', 'system_metrics', 'test_coverage_metrics',
            'performance_metrics', 'resource_usage', 'health_status'
        ]

        for key in required_keys:
            self.assertIn(key, metrics)

        # 检查时间戳
        self.assertIsInstance(metrics['timestamp'], datetime)

        # 检查统计更新
        self.assertEqual(self.collector.collection_stats['total_collections'], 1)
        self.assertEqual(self.collector.collection_stats['successful_collections'], 1)

    def test_system_metrics_collection(self):
        """测试系统指标收集"""
        system_metrics = self.collector._collect_system_metrics()

        # 检查CPU信息
        self.assertIn('cpu', system_metrics)
        cpu_info = system_metrics['cpu']
        self.assertIn('usage_percent', cpu_info)
        self.assertIn('count', cpu_info)
        self.assertIn('count_logical', cpu_info)

        # 检查内存信息
        self.assertIn('memory', system_metrics)
        memory_info = system_metrics['memory']
        self.assertIn('usage_percent', memory_info)
        self.assertIn('total_bytes', memory_info)
        self.assertIn('used_bytes', memory_info)

        # 检查磁盘信息
        self.assertIn('disk', system_metrics)
        disk_info = system_metrics['disk']
        self.assertIn('usage_percent', disk_info)
        self.assertIn('total_bytes', disk_info)

        # 检查网络信息
        self.assertIn('network', system_metrics)
        network_info = system_metrics['network']
        self.assertIn('bytes_sent', network_info)
        self.assertIn('bytes_received', network_info)

    def test_test_coverage_metrics_collection(self):
        """测试测试覆盖率指标收集"""
        coverage_metrics = self.collector._collect_test_coverage_metrics()

        # 即使没有实际文件，也应该返回模拟数据
        self.assertIsInstance(coverage_metrics, dict)

        # 如果有覆盖率相关的键，检查其合理性
        if coverage_metrics:
            if 'overall_coverage' in coverage_metrics:
                coverage = coverage_metrics['overall_coverage']
                self.assertIsInstance(coverage, (int, float))
                self.assertGreaterEqual(coverage, 0)
                self.assertLessEqual(coverage, 100)

    def test_performance_metrics_collection(self):
        """测试性能指标收集"""
        perf_metrics = self.collector._collect_performance_metrics()

        # 检查基本结构
        self.assertIsInstance(perf_metrics, dict)

        # 如果有进程信息，检查其合理性
        if 'process' in perf_metrics:
            process_info = perf_metrics['process']
            self.assertIn('cpu_percent', process_info)

    def test_cache_mechanism(self):
        """测试缓存机制"""
        # 第一次收集
        start_time = time.time()
        metrics1 = self.collector.collect_all_metrics()
        first_duration = time.time() - start_time

        # 立即再次收集（应该使用缓存）
        start_time = time.time()
        metrics2 = self.collector.collect_all_metrics()
        second_duration = time.time() - start_time

        # 缓存访问应该更快
        self.assertLess(second_duration, first_duration)

        # 结果应该相同
        self.assertEqual(metrics1['system_metrics'], metrics2['system_metrics'])

        # 检查缓存统计
        cache_stats = self.collector.get_cache_stats()
        self.assertGreaterEqual(cache_stats['cache_entries'], 1)

    def test_cache_expiration(self):
        """测试缓存过期"""
        # 设置很短的缓存超时
        self.collector.set_cache_timeout(1)

        # 收集数据
        metrics1 = self.collector.collect_all_metrics()
        cache_stats = self.collector.get_cache_stats()

        # 等待缓存过期
        time.sleep(1.1)

        # 再次收集，应该重新获取数据
        metrics2 = self.collector.collect_all_metrics()

        # 时间戳应该不同（因为重新收集了）
        self.assertNotEqual(metrics1['timestamp'], metrics2['timestamp'])

    def test_clear_cache(self):
        """测试清空缓存"""
        # 先收集一些数据
        self.collector.collect_all_metrics()
        cache_stats = self.collector.get_cache_stats()
        initial_entries = cache_stats['cache_entries']

        # 清空缓存
        self.collector.clear_cache()

        # 检查缓存已清空
        cache_stats_after = self.collector.get_cache_stats()
        self.assertEqual(cache_stats_after['cache_entries'], 0)

    def test_get_collection_stats(self):
        """测试获取收集统计"""
        # 收集一些数据
        self.collector.collect_all_metrics()
        self.collector.collect_all_metrics()

        stats = self.collector.get_collection_stats()

        required_keys = [
            'total_collections', 'successful_collections',
            'failed_collections', 'success_rate'
        ]

        for key in required_keys:
            self.assertIn(key, stats)

        self.assertEqual(stats['total_collections'], 2)
        self.assertEqual(stats['successful_collections'], 2)
        self.assertEqual(stats['failed_collections'], 0)
        self.assertEqual(stats['success_rate'], 100.0)

    def test_reset_stats(self):
        """测试重置统计"""
        # 先收集一些数据
        self.collector.collect_all_metrics()
        self.assertEqual(self.collector.collection_stats['total_collections'], 1)

        # 重置统计
        self.collector.reset_stats()

        # 检查统计已重置
        self.assertEqual(self.collector.collection_stats['total_collections'], 0)
        self.assertEqual(self.collector.collection_stats['successful_collections'], 0)

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_system_metrics_with_mocked_psutil(self, mock_net, mock_disk, mock_memory, mock_cpu):
        """测试系统指标收集（使用模拟的psutil）"""
        # 设置模拟返回值
        mock_cpu.return_value = 45.5
        mock_memory.return_value = Mock(
            percent=60.0,
            used=6144000000,
            total=10240000000,
            available=4096000000,
            free=2048000000
        )
        mock_disk.return_value = Mock(
            percent=75.0,
            used=750000000000,
            total=1000000000000,
            free=250000000000
        )
        mock_net.return_value = Mock(
            bytes_sent=1000000,
            bytes_recv=2000000,
            packets_sent=50000,
            packets_recv=75000,
            errin=0,
            errout=0
        )

        metrics = self.collector._collect_system_metrics()

        # 验证结果
        self.assertEqual(metrics['cpu']['usage_percent'], 45.5)
        self.assertEqual(metrics['memory']['usage_percent'], 60.0)
        self.assertEqual(metrics['disk']['usage_percent'], 75.0)
        self.assertEqual(metrics['network']['bytes_sent'], 1000000)

    def test_error_handling(self):
        """测试错误处理"""
        # 模拟异常
        with patch.object(self.collector, '_collect_system_metrics', side_effect=Exception("Test error")):
            metrics = self.collector.collect_all_metrics()

            # 应该返回错误结构
            self.assertIn('error', metrics)
            self.assertEqual(metrics['error'], "Test error")

            # 统计应该反映失败
            self.assertEqual(self.collector.collection_stats['failed_collections'], 1)

    def test_cache_stats_structure(self):
        """测试缓存统计结构"""
        # 先收集一些数据
        self.collector.collect_all_metrics()

        stats = self.collector.get_cache_stats()

        required_keys = ['cache_entries', 'cache_timeout', 'entries']
        for key in required_keys:
            self.assertIn(key, stats)

        # 检查entries结构
        entries = stats['entries']
        self.assertIsInstance(entries, dict)

        # 如果有条目，检查每个条目的结构
        for cache_key, entry_stats in entries.items():
            required_entry_keys = ['age_seconds', 'is_valid', 'expires_in']
            for key in required_entry_keys:
                self.assertIn(key, entry_stats)


if __name__ == '__main__':
    unittest.main()
