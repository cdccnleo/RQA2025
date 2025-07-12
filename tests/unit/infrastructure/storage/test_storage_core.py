import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date
from src.infrastructure.storage.core import QuoteStorage, StorageAdapter
from src.infrastructure.monitoring import StorageMonitor

class MockStorageAdapter(StorageAdapter):
    """Mock存储适配器用于测试"""
    
    def __init__(self):
        self.stored_data = {}
        self.write_called = False
        self.read_called = False
    
    def write(self, path: str, data: dict) -> bool:
        self.write_called = True
        self.stored_data[path] = data
        return True
    
    def read(self, path: str):
        self.read_called = True
        return self.stored_data.get(path)

@pytest.fixture
def mock_adapter():
    """创建mock存储适配器"""
    return MockStorageAdapter()

@pytest.fixture
def quote_storage(mock_adapter):
    """创建QuoteStorage实例"""
    return QuoteStorage(mock_adapter)

@pytest.fixture
def mock_monitor():
    """创建mock监控器"""
    monitor = Mock()
    monitor.record_write = Mock()
    monitor.record_error = Mock()
    return monitor

class TestQuoteStorage:
    """QuoteStorage测试类"""

    def test_init(self, mock_adapter):
        """测试初始化"""
        storage = QuoteStorage(mock_adapter)
        assert storage._adapter == mock_adapter
        assert storage._lock is not None
        assert storage._monitor is not None
        assert 'morning' in storage._trading_hours
        assert 'afternoon' in storage._trading_hours

    def test_save_quote_success(self, quote_storage, mock_adapter):
        """测试成功保存行情数据"""
        data = {
            "time": "09:30:00",
            "price": 1720.5,
            "volume": 1500,
            "limit_status": ""
        }
        
        result = quote_storage.save_quote("600519", data)
        
        assert result is True
        assert mock_adapter.write_called
        assert "quotes/600519" in mock_adapter.stored_data

    def test_save_quote_with_limit_status(self, quote_storage, mock_adapter):
        """测试保存带涨跌停状态的行情数据"""
        data = {
            "time": "09:30:00",
            "price": 1720.5,
            "volume": 1500,
            "limit_status": "up"
        }
        
        result = quote_storage.save_quote("600519", data)
        
        assert result is True
        # 验证涨跌停状态被记录
        assert "limit_status/600519" in mock_adapter.stored_data
        assert mock_adapter.stored_data["limit_status/600519"]["status"] == "up"

    def test_save_quote_exception(self, mock_adapter):
        """测试保存行情数据时发生异常"""
        mock_adapter.write.side_effect = Exception("Storage error")
        storage = QuoteStorage(mock_adapter)
        
        data = {"time": "09:30:00", "price": 1720.5}
        
        with pytest.raises(Exception, match="Storage error"):
            storage.save_quote("600519", data)

    def test_get_quote_success(self, quote_storage, mock_adapter):
        """测试成功读取行情数据"""
        expected_data = {"time": "09:30:00", "price": 1720.5}
        mock_adapter.stored_data["quotes/600519/2024-01-01"] = expected_data
        
        result = quote_storage.get_quote("600519", "2024-01-01")
        
        assert result == expected_data
        assert mock_adapter.read_called

    def test_get_quote_not_found(self, quote_storage, mock_adapter):
        """测试读取不存在的行情数据"""
        result = quote_storage.get_quote("600519", "2024-01-01")
        
        assert result is None
        assert mock_adapter.read_called

    @patch('src.infrastructure.storage.core.StorageMonitor')
    def test_monitor_integration(self, mock_monitor_class, mock_adapter):
        """测试监控器集成"""
        mock_monitor = Mock()
        mock_monitor_class.return_value = mock_monitor
        storage = QuoteStorage(mock_adapter)
        
        data = {"time": "09:30:00", "price": 1720.5}
        storage.save_quote("600519", data)
        
        mock_monitor.record_write.assert_called_once()

    def test_thread_safety(self, mock_adapter):
        """测试线程安全性"""
        storage = QuoteStorage(mock_adapter)
        import threading
        import time
        
        results = []
        
        def save_quote_thread():
            data = {"time": "09:30:00", "price": 1720.5}
            result = storage.save_quote("600519", data)
            results.append(result)
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=save_quote_thread)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        assert all(result is True for result in results)

    def test_trading_hours_config(self, mock_adapter):
        """测试交易时间配置"""
        storage = QuoteStorage(mock_adapter)
        
        assert storage._trading_hours['morning'] == (9*60+30, 11*60+30)
        assert storage._trading_hours['afternoon'] == (13*60, 15*60)

class TestStorageAdapter:
    """StorageAdapter测试类"""

    def test_write_not_implemented(self):
        """测试write方法未实现"""
        adapter = StorageAdapter()
        with pytest.raises(NotImplementedError):
            adapter.write("test_path", {"data": "test"})

    def test_read_not_implemented(self):
        """测试read方法未实现"""
        adapter = StorageAdapter()
        with pytest.raises(NotImplementedError):
            adapter.read("test_path")

class TestMockStorageAdapter:
    """MockStorageAdapter测试类"""

    def test_write_implementation(self):
        """测试MockStorageAdapter的write实现"""
        adapter = MockStorageAdapter()
        data = {"test": "data"}
        
        result = adapter.write("test_path", data)
        
        assert result is True
        assert adapter.write_called
        assert adapter.stored_data["test_path"] == data

    def test_read_implementation(self):
        """测试MockStorageAdapter的read实现"""
        adapter = MockStorageAdapter()
        expected_data = {"test": "data"}
        adapter.stored_data["test_path"] = expected_data
        
        result = adapter.read("test_path")
        
        assert result == expected_data
        assert adapter.read_called

    def test_read_not_found(self):
        """测试MockStorageAdapter读取不存在的数据"""
        adapter = MockStorageAdapter()
        
        result = adapter.read("nonexistent_path")
        
        assert result is None
        assert adapter.read_called 