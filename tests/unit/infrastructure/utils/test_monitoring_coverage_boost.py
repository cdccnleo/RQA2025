"""
基础设施层监控模块覆盖率提升测试
测试日期: 2025-12-19
目标: 提升utils/monitoring模块覆盖率至70%+
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta


class TestMonitoringCoverageBoost:
    """监控模块覆盖率提升测试"""

    def setup_method(self):
        """测试前准备"""
        self.mock_logger = Mock()

    def test_log_backpressure_plugin_initialization(self):
        """测试日志背压插件初始化"""
        from src.infrastructure.utils.monitoring.log_backpressure_plugin import LogBackpressurePlugin

        # 测试正常初始化
        plugin = LogBackpressurePlugin(max_queue_size=1000, threshold=0.8)
        assert plugin.max_queue_size == 1000
        assert plugin.threshold == 0.8
        assert plugin.current_size == 0

        # 测试默认参数
        plugin_default = LogBackpressurePlugin()
        assert plugin_default.max_queue_size == 10000
        assert plugin_default.threshold == 0.9

    def test_log_backpressure_plugin_monitoring(self):
        """测试日志背压监控功能"""
        from src.infrastructure.utils.monitoring.log_backpressure_plugin import LogBackpressurePlugin

        plugin = LogBackpressurePlugin(max_queue_size=100, threshold=0.8)

        # 测试正常监控
        plugin.update_queue_size(50)
        assert plugin.current_size == 50
        assert not plugin.is_backpressured()

        # 测试背压状态
        plugin.update_queue_size(90)
        assert plugin.current_size == 90
        assert plugin.is_backpressured()

        # 测试阈值计算
        assert plugin.get_backpressure_ratio() == 0.9

    def test_log_backpressure_plugin_actions(self):
        """测试背压处理动作"""
        from src.infrastructure.utils.monitoring.log_backpressure_plugin import LogBackpressurePlugin

        plugin = LogBackpressurePlugin(max_queue_size=100, threshold=0.8)

        # 测试背压缓解
        plugin.update_queue_size(90)
        assert plugin.is_backpressured()

        plugin.relieve_backpressure(30)
        assert plugin.current_size == 60
        assert not plugin.is_backpressured()

    def test_log_compressor_plugin_initialization(self):
        """测试日志压缩插件初始化"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin

        plugin = LogCompressorPlugin(compression_level=6, max_file_size=1048576)
        assert plugin.compression_level == 6
        assert plugin.max_file_size == 1048576
        assert plugin.compressed_files == []

    def test_log_compressor_plugin_compression(self):
        """测试日志压缩功能"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin

        plugin = LogCompressorPlugin()

        # 模拟日志文件
        test_content = "Test log content\n" * 1000

        # 测试压缩（这里只是模拟，实际压缩需要文件系统操作）
        with patch('gzip.open') as mock_gzip:
            with patch('os.path.exists', return_value=True):
                with patch('os.remove'):
                    with patch('os.rename'):
                        result = plugin.compress_log_file("/path/to/log.txt")
                        assert result is True

    def test_log_compressor_plugin_cleanup(self):
        """测试压缩文件清理"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin

        plugin = LogCompressorPlugin(retention_days=7)

        # 模拟旧文件清理
        with patch('os.listdir', return_value=['log1.gz', 'log2.gz']):
            with patch('os.path.getctime') as mock_getctime:
                # 模拟一个文件是10天前，一个是1天前
                mock_getctime.side_effect = lambda f: time.time() - (10 * 24 * 3600 if 'log1' in f else 24 * 3600)
                with patch('os.remove') as mock_remove:
                    plugin.cleanup_old_files("/path/to/logs")
                    # 应该只删除10天的文件
                    assert mock_remove.call_count == 1

    def test_market_data_logger_initialization(self):
        """测试市场数据日志器初始化"""
        from src.infrastructure.utils.monitoring.market_data_logger import MarketDataLogger

        logger = MarketDataLogger(log_dir="/tmp/logs", max_file_size=1024*1024)
        assert logger.log_dir == "/tmp/logs"
        assert logger.max_file_size == 1024*1024
        assert logger.current_file_size == 0

    def test_market_data_logger_logging(self):
        """测试市场数据日志记录"""
        from src.infrastructure.utils.monitoring.market_data_logger import MarketDataLogger

        logger = MarketDataLogger()

        # 测试数据记录
        test_data = {
            'symbol': '000001.SZ',
            'price': 10.5,
            'volume': 1000,
            'timestamp': datetime.now()
        }

        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            logger.log_market_data(test_data)
            mock_file.write.assert_called()

    def test_market_data_logger_rotation(self):
        """测试日志轮转"""
        from src.infrastructure.utils.monitoring.market_data_logger import MarketDataLogger

        logger = MarketDataLogger(max_file_size=100)

        with patch('os.path.getsize', return_value=150):
            with patch('os.rename') as mock_rename:
                logger._check_rotation("test.log")
                mock_rename.assert_called_once()

    def test_storage_monitor_plugin_initialization(self):
        """测试存储监控插件初始化"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin

        plugin = StorageMonitorPlugin(threshold=0.85, check_interval=300)
        assert plugin.threshold == 0.85
        assert plugin.check_interval == 300
        assert plugin.last_check == 0

    def test_storage_monitor_plugin_monitoring(self):
        """测试存储监控"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin

        plugin = StorageMonitorPlugin(threshold=0.8)

        # 模拟存储使用情况
        with patch('shutil.disk_usage') as mock_disk:
            # 模拟总空间100GB，已用90GB
            mock_disk.return_value = Mock(total=100*1024**3, used=90*1024**3, free=10*1024**3)

            usage = plugin.check_storage_usage("/tmp")
            assert usage['total'] == 100*1024**3
            assert usage['used'] == 90*1024**3
            assert usage['usage_ratio'] == 0.9
            assert plugin.is_storage_critical("/tmp")

    def test_storage_monitor_plugin_alerts(self):
        """测试存储告警"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin

        plugin = StorageMonitorPlugin(threshold=0.8)

        # 测试告警生成
        with patch('shutil.disk_usage') as mock_disk:
            mock_disk.return_value = Mock(total=100*1024**3, used=95*1024**3, free=5*1024**3)

            alerts = plugin.get_storage_alerts("/tmp")
            assert len(alerts) > 0
            assert "critical" in alerts[0]['level'].lower()

    def test_monitoring_integration(self):
        """测试监控组件集成"""
        from src.infrastructure.utils.monitoring.log_backpressure_plugin import LogBackpressurePlugin
        from src.infrastructure.utils.monitoring.market_data_logger import MarketDataLogger

        # 测试组件协作
        backpressure_plugin = LogBackpressurePlugin(max_queue_size=1000)
        market_logger = MarketDataLogger()

        # 模拟监控场景
        backpressure_plugin.update_queue_size(500)
        assert not backpressure_plugin.is_backpressured()

        backpressure_plugin.update_queue_size(950)
        assert backpressure_plugin.is_backpressured()

        # 测试日志记录
        test_data = {'symbol': 'AAPL', 'price': 150.0}
        with patch('builtins.open', create=True):
            market_logger.log_market_data(test_data)

    def test_monitoring_error_handling(self):
        """测试监控组件错误处理"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin

        plugin = LogCompressorPlugin()

        # 测试异常处理
        with patch('gzip.open', side_effect=Exception("Compression failed")):
            result = plugin.compress_log_file("/nonexistent/file.log")
            assert result is False

    def test_monitoring_performance(self):
        """测试监控组件性能"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin

        plugin = StorageMonitorPlugin()

        start_time = time.time()
        for _ in range(100):
            with patch('shutil.disk_usage', return_value=Mock(total=100, used=50, free=50)):
                plugin.check_storage_usage("/tmp")
        end_time = time.time()

        # 确保性能在合理范围内
        assert end_time - start_time < 1.0  # 100次检查应该在1秒内完成
























