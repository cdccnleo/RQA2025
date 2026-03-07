#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层日志压缩插件组件测试

测试目标：提升utils/monitoring/log_compressor_plugin.py的真实覆盖率
实际导入和使用src.infrastructure.utils.monitoring.log_compressor_plugin模块
"""

import pytest
import tempfile
import os
from unittest.mock import patch


class TestLogCompressorPlugin:
    """测试日志压缩插件类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        config = {
            "algorithm": "zstd",
            "level": 3,
            "chunk_size": 1048576,
            "thread_safe": True
        }
        
        plugin = LogCompressorPlugin(config)
        
        assert plugin.config == config
        assert plugin.chunk_size == 1048576
        assert plugin.lock is not None
    
    def test_init_default(self):
        """测试使用默认配置初始化"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        # LogCompressorPlugin需要config参数，不能为None
        config = {}
        plugin = LogCompressorPlugin(config)
        
        assert plugin.chunk_size > 0
        assert plugin.strategy == "default"
    
    def test_compress(self):
        """测试压缩数据"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        config = {"algorithm": "zstd", "level": 3}
        plugin = LogCompressorPlugin(config)
        data = b"test data to compress" * 100
        
        compressed = plugin.compress(data)
        
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
    
    def test_decompress(self):
        """测试解压缩数据"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        config = {"algorithm": "zstd", "level": 3}
        plugin = LogCompressorPlugin(config)
        original_data = b"test data to compress" * 100
        compressed = plugin.compress(original_data)
        
        decompressed = plugin.decompress(compressed)
        
        assert decompressed == original_data
    
    def test_stream_compress(self):
        """测试流式压缩文件"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        config = {"algorithm": "zstd", "level": 3}
        plugin = LogCompressorPlugin(config)
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f_in:
            f_in.write(b"test data" * 1000)
            input_file = f_in.name
        
        output_file = input_file + ".zst"
        
        try:
            plugin.stream_compress(input_file, output_file)
            
            assert os.path.exists(output_file)
            assert os.path.getsize(output_file) > 0
        finally:
            if os.path.exists(input_file):
                os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_should_compress(self):
        """测试判断是否应该压缩"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        config = {}
        plugin = LogCompressorPlugin(config)
        
        result = plugin.should_compress()
        
        assert isinstance(result, bool)
    
    def test_auto_select_strategy(self):
        """测试自动选择策略"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        config = {}
        plugin = LogCompressorPlugin(config)
        
        with patch('psutil.cpu_percent', return_value=80):
            strategy = plugin.auto_select_strategy()
            assert strategy in ["light", "aggressive", "default"]
            assert plugin.current_strategy in ["light", "aggressive", "default"]
    
    def test_get_compression_stats(self):
        """测试获取压缩统计信息"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        config = {}
        plugin = LogCompressorPlugin(config)
        
        stats = plugin.get_compression_stats()
        
        assert isinstance(stats, dict)
        assert "algorithm" in stats
        assert "level" in stats
        assert "chunk_size" in stats
    
    def test_get_supported_algorithms(self):
        """测试获取支持的压缩算法"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        config = {}
        plugin = LogCompressorPlugin(config)
        
        algorithms = plugin.get_supported_algorithms()
        
        assert isinstance(algorithms, list)
        assert "zstd" in algorithms
    
    def test_is_compression_effective(self):
        """测试判断压缩是否有效"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        config = {}
        plugin = LogCompressorPlugin(config)
        
        # 测试有效压缩
        assert plugin.is_compression_effective(1000, 500) is True
        
        # 测试无效压缩
        assert plugin.is_compression_effective(1000, 950) is False
    
    def test_update_strategy(self):
        """测试更新压缩策略"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        config = {}
        plugin = LogCompressorPlugin(config)
        plugin.update_strategy("aggressive")
        
        assert plugin.strategy == "aggressive"
        assert plugin.current_strategy == "aggressive"
    
    def test_validate_config(self):
        """测试验证配置"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import LogCompressorPlugin
        
        config = {}
        plugin = LogCompressorPlugin(config)
        
        valid_config = {
            "algorithm": "zstd",
            "level": 3,
            "chunk_size": 1024
        }
        
        assert plugin.validate_config(valid_config) is True
        
        invalid_config = {"algorithm": "zstd"}
        assert plugin.validate_config(invalid_config) is False


class TestTradingHoursAwareCompressor:
    """测试交易时段感知的智能压缩器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import TradingHoursAwareCompressor
        
        config = {
            "algorithm": "zstd",
            "level": 3
        }
        
        compressor = TradingHoursAwareCompressor(config)
        
        assert compressor.trading_hours is not None
        assert "morning" in compressor.trading_hours
    
    def test_should_compress(self):
        """测试根据交易时段决定压缩策略"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import TradingHoursAwareCompressor
        
        config = {"algorithm": "zstd", "level": 3}
        compressor = TradingHoursAwareCompressor(config)
        
        result = compressor.should_compress()
        
        assert isinstance(result, bool)


class TestCompressionManager:
    """测试压缩策略管理器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import CompressionManager
        
        manager = CompressionManager()
        
        assert isinstance(manager.compressors, dict)
        assert manager.current_strategy is None
    
    def test_register_strategy(self):
        """测试注册压缩策略"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import CompressionManager, LogCompressorPlugin
        
        manager = CompressionManager()
        compressor = LogCompressorPlugin({})
        
        manager.register_strategy("test_strategy", compressor)
        
        assert "test_strategy" in manager.compressors
        assert manager.compressors["test_strategy"] == compressor
    
    def test_auto_select_strategy(self):
        """测试根据系统负载自动选择策略"""
        from src.infrastructure.utils.monitoring.log_compressor_plugin import CompressionManager, LogCompressorPlugin
        
        manager = CompressionManager()
        manager.register_strategy("light", LogCompressorPlugin({}))
        manager.register_strategy("aggressive", LogCompressorPlugin({}))
        
        with patch('psutil.cpu_percent', return_value=80):
            manager.auto_select_strategy()
            assert manager.current_strategy is not None

