#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层核心组件测试

测试目标：提升utils/components/core.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.core模块
"""

import pytest
from unittest.mock import MagicMock, Mock
from datetime import datetime
import importlib.util
import os


# 直接导入core.py模块
_core_path = os.path.join(
    os.path.dirname(__file__),
    '../../../../../src/infrastructure/utils/components/core.py'
)
spec = importlib.util.spec_from_file_location("core_module", _core_path)
core_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core_module)

QuoteStorage = core_module.QuoteStorage
StorageAdapter = core_module.StorageAdapter


class TestStorageAdapter:
    """测试存储适配器抽象基类"""
    
    def test_storage_adapter_cannot_instantiate(self):
        """测试存储适配器不能直接实例化"""
        # StorageAdapter可能不是抽象类，尝试实例化
        try:
            adapter = StorageAdapter()
            # 如果可以实例化，测试方法调用
            with pytest.raises(NotImplementedError):
                adapter.write("path", {})
        except TypeError:
            # 如果不能实例化，这是预期的
            pass
    
    def test_storage_adapter_write_not_implemented(self):
        """测试write方法未实现"""
        class ConcreteAdapter(StorageAdapter):
            def read(self, path: str):
                return None
        
        adapter = ConcreteAdapter()
        with pytest.raises(NotImplementedError):
            adapter.write("path", {})
    
    def test_storage_adapter_read_not_implemented(self):
        """测试read方法未实现"""
        class ConcreteAdapter(StorageAdapter):
            def write(self, path: str, data: dict) -> bool:
                return True
        
        adapter = ConcreteAdapter()
        with pytest.raises(NotImplementedError):
            adapter.read("path")


class TestQuoteStorage:
    """测试行情数据存储核心类"""
    
    def test_init(self):
        """测试初始化"""
        class MockAdapter(StorageAdapter):
            def write(self, path: str, data: dict) -> bool:
                return True
            
            def read(self, path: str):
                return None
        
        adapter = MockAdapter()
        storage = QuoteStorage(adapter)
        
        assert storage._adapter == adapter
        assert hasattr(storage, '_lock')
        assert hasattr(storage, '_monitor')
        assert hasattr(storage, '_trading_hours')
    
    def test_save_quote(self):
        """测试存储行情数据"""
        class MockAdapter(StorageAdapter):
            def __init__(self):
                self.written_data = {}
            
            def write(self, path: str, data: dict) -> bool:
                self.written_data[path] = data
                return True
            
            def read(self, path: str):
                return self.written_data.get(path)
        
        adapter = MockAdapter()
        storage = QuoteStorage(adapter)
        
        data = {
            "time": "09:30:00",
            "price": 1720.5,
            "volume": 1500,
            "limit_status": ""
        }
        
        result = storage.save_quote("600519", data)
        assert result is True
    
    def test_save_quote_with_limit_status(self):
        """测试存储带涨跌停状态的行情数据"""
        class MockAdapter(StorageAdapter):
            def __init__(self):
                self.written_data = {}
            
            def write(self, path: str, data: dict) -> bool:
                self.written_data[path] = data
                return True
            
            def read(self, path: str):
                return self.written_data.get(path)
        
        adapter = MockAdapter()
        storage = QuoteStorage(adapter)
        
        data = {
            "time": "09:30:00",
            "price": 1720.5,
            "volume": 1500,
            "limit_status": "up"
        }
        
        result = storage.save_quote("600519", data)
        assert result is True
    
    def test_save_quote_exception(self):
        """测试存储行情数据异常处理"""
        class MockAdapter(StorageAdapter):
            def write(self, path: str, data: dict) -> bool:
                raise Exception("Write failed")
            
            def read(self, path: str):
                return None
        
        adapter = MockAdapter()
        storage = QuoteStorage(adapter)
        
        data = {
            "time": "09:30:00",
            "price": 1720.5,
            "volume": 1500
        }
        
        with pytest.raises(Exception):
            storage.save_quote("600519", data)
    
    def test_get_quote(self):
        """测试读取行情数据"""
        class MockAdapter(StorageAdapter):
            def __init__(self):
                self.data = {
                    "quotes/600519/2024-01-01": {"price": 1720.5}
                }
            
            def write(self, path: str, data: dict) -> bool:
                return True
            
            def read(self, path: str):
                return self.data.get(path)
        
        adapter = MockAdapter()
        storage = QuoteStorage(adapter)
        
        result = storage.get_quote("600519", "2024-01-01")
        assert result is not None
        assert result["price"] == 1720.5
    
    def test_get_quote_not_found(self):
        """测试读取不存在的行情数据"""
        class MockAdapter(StorageAdapter):
            def write(self, path: str, data: dict) -> bool:
                return True
            
            def read(self, path: str):
                return None
        
        adapter = MockAdapter()
        storage = QuoteStorage(adapter)
        
        result = storage.get_quote("600519", "2024-01-01")
        assert result is None

