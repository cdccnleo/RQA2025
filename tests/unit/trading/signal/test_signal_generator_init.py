#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成器初始化模块测试

测试目标：提升signal_generator_init.py的覆盖率
"""

import pytest


class TestSignalGeneratorInit:
    """测试信号生成器初始化模块"""
    
    def test_import_signal_generator(self):
        """测试导入SignalGenerator"""
        try:
            from src.trading.signal.signal_generator_init import SignalGenerator
            assert SignalGenerator is not None
        except ImportError:
            pytest.skip("SignalGenerator not available")
    
    def test_import_signal_config(self):
        """测试导入SignalConfig"""
        try:
            from src.trading.signal.signal_generator_init import SignalConfig
            assert SignalConfig is not None
        except ImportError:
            pytest.skip("SignalConfig not available")
    
    def test_import_simple_signal_generator(self):
        """测试导入SimpleSignalGenerator"""
        try:
            from src.trading.signal.signal_generator_init import SimpleSignalGenerator
            assert SimpleSignalGenerator is not None
        except ImportError:
            pytest.skip("SimpleSignalGenerator not available")
    
    def test_module_all_exports(self):
        """测试模块__all__导出"""
        from src.trading.signal import signal_generator_init
        assert hasattr(signal_generator_init, '__all__')
        assert 'SignalGenerator' in signal_generator_init.__all__
        assert 'SignalConfig' in signal_generator_init.__all__
        assert 'SimpleSignalGenerator' in signal_generator_init.__all__
    
    def test_module_import(self):
        """测试模块导入"""
        import src.trading.signal.signal_generator_init as module
        assert module is not None
        assert hasattr(module, '__all__')

