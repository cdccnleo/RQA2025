#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tempfile
import shutil
import json
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.infrastructure.storage.core import QuoteStorage, StorageAdapter
from src.infrastructure.storage.adapters.file_system import FileSystemAdapter, AShareFileSystemAdapter


class TestStorageAdapter:
    """测试存储适配器基类"""

    def test_storage_adapter_abstract_methods(self):
        """测试抽象方法"""
        adapter = StorageAdapter()
        
        with pytest.raises(NotImplementedError):
            adapter.write("test", {})
        
        with pytest.raises(NotImplementedError):
            adapter.read("test")


class TestFileSystemAdapter:
    """测试文件系统存储适配器"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def adapter(self, temp_dir):
        """文件系统适配器"""
        return FileSystemAdapter(base_path=temp_dir)

    def test_file_system_adapter_init(self, temp_dir):
        """测试初始化"""
        adapter = FileSystemAdapter(base_path=temp_dir)
        assert adapter.base_path == Path(temp_dir)
        assert adapter.base_path.exists()

    def test_write_success(self, adapter):
        """测试成功写入"""
        data = {"test": "data", "number": 123}
        result = adapter.write("test/path", data)
        
        assert result is True
        file_path = adapter.base_path / "test/path.json"
        assert file_path.exists()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        assert saved_data == data

    def test_write_creates_directories(self, adapter):
        """测试写入时创建目录"""
        data = {"test": "data"}
        adapter.write("deep/nested/path", data)
        
        dir_path = adapter.base_path / "deep/nested"
        assert dir_path.exists()

    def test_write_failure(self, adapter):
        """测试写入失败"""
        # 模拟JSON编码错误
        data = {"invalid": object()}  # 不可序列化的对象
        
        result = adapter.write("test/path", data)
        assert result is False

    def test_read_success(self, adapter):
        """测试成功读取"""
        data = {"test": "data", "number": 123}
        adapter.write("test/path", data)
        
        result = adapter.read("test/path")
        assert result == data

    def test_read_file_not_exists(self, adapter):
        """测试读取不存在的文件"""
        result = adapter.read("nonexistent/path")
        assert result is None

    def test_read_invalid_json(self, adapter):
        """测试读取无效JSON"""
        # 创建无效JSON文件
        file_path = adapter.base_path / "invalid.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("invalid json content")
        
        result = adapter.read("invalid")
        assert result is None

    def test_read_io_error(self, adapter):
        """测试读取IO错误"""
        # 模拟文件权限问题
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            result = adapter.read("test/path")
            assert result is None


class TestAShareFileSystemAdapter:
    """测试A股专用文件存储适配器"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def adapter(self, temp_dir):
        """A股文件系统适配器"""
        return AShareFileSystemAdapter(base_path=temp_dir)

    def test_save_quote_sh_stock(self, adapter):
        """测试存储上海股票"""
        data = {"price": 100.0, "volume": 1000}
        result = adapter.save_quote("600519", data)
        
        assert result is True
        assert data["market"] == "SH"
        
        # 验证文件存在
        file_path = adapter.base_path / "quotes/600519.json"
        assert file_path.exists()

    def test_save_quote_sz_stock(self, adapter):
        """测试存储深圳股票"""
        data = {"price": 200.0, "volume": 2000}
        result = adapter.save_quote("000001", data)
        
        assert result is True
        assert data["market"] == "SZ"

    def test_save_quote_gem_stock(self, adapter):
        """测试存储创业板股票"""
        data = {"price": 50.0, "volume": 500}
        result = adapter.save_quote("300001", data)
        
        assert result is True
        assert data["market"] == "SH"

    def test_save_quote_invalid_symbol(self, adapter):
        """测试无效股票代码"""
        data = {"price": 100.0}
        
        with pytest.raises(ValueError, match="非A股股票代码"):
            adapter.save_quote("AAPL", data)

    def test_save_quote_write_failure(self, adapter):
        """测试写入失败"""
        # 模拟写入失败
        with patch.object(adapter, 'write', return_value=False):
            data = {"price": 100.0}
            result = adapter.save_quote("600519", data)
            assert result is False


class TestQuoteStorage:
    """测试行情数据存储"""

    @pytest.fixture
    def mock_adapter(self):
        """模拟存储适配器"""
        adapter = Mock(spec=StorageAdapter)
        adapter.write.return_value = True
        adapter.read.return_value = {"test": "data"}
        return adapter

    @pytest.fixture
    def quote_storage(self, mock_adapter):
        """行情存储实例"""
        return QuoteStorage(mock_adapter)

    def test_quote_storage_init(self, mock_adapter):
        """测试初始化"""
        storage = QuoteStorage(mock_adapter)
        assert storage._adapter == mock_adapter
        assert storage._lock is not None
        assert storage._monitor is not None

    def test_save_quote_success(self, quote_storage, mock_adapter):
        """测试成功保存行情"""
        data = {
            "time": "09:30:00",
            "price": 1720.5,
            "volume": 1500,
            "limit_status": ""
        }
        
        result = quote_storage.save_quote("600519", data)
        
        assert result is True
        mock_adapter.write.assert_called_once()
        
        # 验证调用参数
        call_args = mock_adapter.write.call_args
        assert "quotes/600519" in call_args[0][0]  # 路径包含股票代码

    def test_save_quote_with_limit_status(self, quote_storage, mock_adapter):
        """测试保存带涨跌停状态的行情"""
        data = {
            "time": "09:30:00",
            "price": 1720.5,
            "volume": 1500,
            "limit_status": "up"
        }
        
        quote_storage.save_quote("600519", data)
        
        # 验证调用了两次write：一次保存行情，一次保存涨跌停状态
        assert mock_adapter.write.call_count == 2
        
        # 验证涨跌停状态保存
        calls = mock_adapter.write.call_args_list
        limit_status_call = next(call for call in calls if "limit_status" in call[0][0])
        assert limit_status_call is not None

    def test_save_quote_exception(self, quote_storage, mock_adapter):
        """测试保存行情异常"""
        mock_adapter.write.side_effect = Exception("Storage error")
        
        data = {"time": "09:30:00", "price": 1720.5}
        
        with pytest.raises(Exception, match="Storage error"):
            quote_storage.save_quote("600519", data)

    def test_get_quote_success(self, quote_storage, mock_adapter):
        """测试成功获取行情"""
        expected_data = {"time": "09:30:00", "price": 1720.5}
        mock_adapter.read.return_value = expected_data
        
        result = quote_storage.get_quote("600519", "2023-01-01")
        
        assert result == expected_data
        mock_adapter.read.assert_called_once_with("quotes/600519/2023-01-01")

    def test_get_quote_not_found(self, quote_storage, mock_adapter):
        """测试获取不存在的行情"""
        mock_adapter.read.return_value = None
        
        result = quote_storage.get_quote("600519", "2023-01-01")
        
        assert result is None

    def test_thread_safety(self, quote_storage):
        """测试线程安全性"""
        import threading
        import time
        
        results = []
        errors = []
        
        def save_quote():
            try:
                data = {"time": "09:30:00", "price": 100.0}
                result = quote_storage.save_quote("600519", data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程同时保存
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=save_quote)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证没有异常且所有操作都成功
        assert len(errors) == 0
        assert len(results) == 10
        assert all(results)

    def test_monitor_integration(self, quote_storage, mock_adapter):
        """测试监控集成"""
        data = {"time": "09:30:00", "price": 1720.5}
        
        quote_storage.save_quote("600519", data)
        
        # 验证监控记录
        assert quote_storage._monitor is not None

    def test_trading_hours_config(self, quote_storage):
        """测试交易时间配置"""
        expected_hours = {
            'morning': (9*60+30, 11*60+30),  # 9:30-11:30
            'afternoon': (13*60, 15*60)      # 13:00-15:00
        }
        
        assert quote_storage._trading_hours == expected_hours

    def test_set_limit_status_internal(self, quote_storage, mock_adapter):
        """测试内部涨跌停状态设置"""
        quote_storage._set_limit_status("600519", "up")
        
        mock_adapter.write.assert_called_once()
        call_args = mock_adapter.write.call_args
        assert "limit_status/600519" in call_args[0][0]
        
        # 验证数据格式
        data = call_args[0][1]
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "up"


class TestStorageIntegration:
    """测试存储集成"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_quote_storage_with_file_system(self, temp_dir):
        """测试行情存储与文件系统集成"""
        adapter = FileSystemAdapter(base_path=temp_dir)
        storage = QuoteStorage(adapter)
        
        # 保存行情
        data = {
            "time": "09:30:00",
            "price": 1720.5,
            "volume": 1500,
            "limit_status": "up"
        }
        
        result = storage.save_quote("600519", data)
        assert result is True
        
        # 验证文件存在
        today = datetime.now().date()
        quote_file = Path(temp_dir) / f"quotes/600519/{today}.json"
        limit_file = Path(temp_dir) / "limit_status/600519.json"
        
        assert quote_file.exists()
        assert limit_file.exists()
        
        # 验证数据内容
        with open(quote_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        assert saved_data == data
        
        # 验证涨跌停状态
        with open(limit_file, 'r', encoding='utf-8') as f:
            limit_data = json.load(f)
        assert limit_data["status"] == "up"

    def test_ashare_adapter_integration(self, temp_dir):
        """测试A股适配器集成"""
        adapter = AShareFileSystemAdapter(base_path=temp_dir)
        
        # 保存行情
        data = {"price": 100.0, "volume": 1000}
        result = adapter.save_quote("600519", data)
        
        assert result is True
        assert data["market"] == "SH"
        
        # 验证文件内容
        file_path = Path(temp_dir) / "quotes/600519.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        assert saved_data["market"] == "SH"
        assert saved_data["price"] == 100.0

    def test_concurrent_access(self, temp_dir):
        """测试并发访问"""
        adapter = FileSystemAdapter(base_path=temp_dir)
        storage = QuoteStorage(adapter)
        
        import threading
        import time
        
        def save_quotes():
            for i in range(10):
                data = {
                    "time": f"09:{i:02d}:00",
                    "price": 100.0 + i,
                    "volume": 1000 + i
                }
                storage.save_quote(f"60051{i}", data)
                time.sleep(0.01)
        
        # 创建多个线程
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=save_quotes)
            threads.append(thread)
            thread.start()
        
        # 等待完成
        for thread in threads:
            thread.join()
        
        # 验证所有文件都创建成功
        today = datetime.now().date()
        for i in range(10):
            file_path = Path(temp_dir) / f"quotes/60051{i}/{today}.json"
            assert file_path.exists()

    def test_error_recovery(self, temp_dir):
        """测试错误恢复"""
        adapter = FileSystemAdapter(base_path=temp_dir)
        storage = QuoteStorage(adapter)
        
        # 正常保存
        data = {"time": "09:30:00", "price": 100.0}
        result = storage.save_quote("600519", data)
        assert result is True
        
        # 模拟存储失败
        with patch.object(adapter, 'write', return_value=False):
            result = storage.save_quote("600520", data)
            assert result is False
        
        # 再次正常保存
        with patch.object(adapter, 'write', return_value=True):
            result = storage.save_quote("600521", data)
            assert result is True 