#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层utils/monitoring/__init__.py模块测试

测试目标：提升utils/monitoring/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.utils.monitoring模块
"""

import pytest


class TestMonitoringInit:
    """测试monitoring模块初始化"""
    
    def test_get_logger_import(self):
        """测试get_logger函数导入"""
        from src.infrastructure.utils.monitoring import get_logger
        
        assert callable(get_logger)
    
    def test_market_data_deduplicator_import(self):
        """测试MarketDataDeduplicator导入"""
        from src.infrastructure.utils.monitoring import MarketDataDeduplicator
        
        assert MarketDataDeduplicator is not None
    
    def test_adaptive_backpressure_plugin_import(self):
        """测试AdaptiveBackpressurePlugin导入"""
        from src.infrastructure.utils.monitoring import AdaptiveBackpressurePlugin
        
        assert AdaptiveBackpressurePlugin is not None
    
    def test_backpressure_handler_plugin_import(self):
        """测试BackpressureHandlerPlugin导入"""
        from src.infrastructure.utils.monitoring import BackpressureHandlerPlugin
        
        assert BackpressureHandlerPlugin is not None
    
    def test_log_compressor_plugin_import(self):
        """测试LogCompressorPlugin导入"""
        from src.infrastructure.utils.monitoring import LogCompressorPlugin
        
        assert LogCompressorPlugin is not None
    
    def test_storage_monitor_plugin_import(self):
        """测试StorageMonitorPlugin导入"""
        from src.infrastructure.utils.monitoring import StorageMonitorPlugin
        
        assert StorageMonitorPlugin is not None
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.utils.monitoring import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "get_logger" in __all__
        assert "MarketDataDeduplicator" in __all__
        assert "StorageMonitorPlugin" in __all__

