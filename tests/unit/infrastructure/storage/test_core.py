#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import unittest
import tempfile
import shutil
import os
from datetime import datetime
from unittest.mock import MagicMock
from src.infrastructure.storage.core import QuoteStorage, StorageAdapter

class TestStorageAdapter(StorageAdapter):
    """测试用存储适配器"""
    
    def __init__(self):
        self.data = {}
        
    def write(self, path: str, data: dict) -> bool:
        self.data[path] = data
        return True
        
    def read(self, path: str) -> Optional[dict]:
        return self.data.get(path)

class TestQuoteStorage(unittest.TestCase):
    """QuoteStorage单元测试"""
    
    def setUp(self):
        import tempfile
        from src.infrastructure.storage.adapters.file_system import FileSystemAdapter
        
        self.temp_dir = tempfile.mkdtemp()
        self.adapter = FileSystemAdapter(self.temp_dir)
        self.storage = QuoteStorage(self.adapter)
        
    def test_save_quote(self):
        """测试行情数据存储"""
        test_data = {
            "time": "09:30:00",
            "price": 1720.5,
            "volume": 1500
        }
        
        result = self.storage.save_quote("600519", test_data)
        self.assertTrue(result)
        
        # 验证数据路径格式
        today = datetime.now().date().isoformat()
        expected_path = f"quotes/600519/{today}"
        
        # 验证文件是否存在
        expected_file = os.path.join(self.temp_dir, f"{expected_path}.json")
        self.assertTrue(os.path.exists(expected_file))
        
    def test_limit_status(self):
        """测试涨跌停状态记录"""
        test_data = {
            "time": "09:30:00",
            "price": 1720.5,
            "volume": 1500,
            "limit_status": "up"
        }
        
        self.storage.save_quote("600519", test_data)
        
        # 验证状态文件是否存在
        status_file = os.path.join(self.temp_dir, "limit_status/600519.json")
        self.assertTrue(os.path.exists(status_file))
        
    def test_thread_safety(self):
        """测试线程安全性"""
        from threading import Thread
        
        results = []
        def worker():
            test_data = {
                "time": "09:30:00",
                "price": 1720.5,
                "volume": 1500
            }
            results.append(
                self.storage.save_quote("600519", test_data)
            )
            
        threads = [Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        self.assertTrue(all(results))
        
        # 验证文件数量（应该有多个文件）
        quote_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.json')]
        self.assertGreater(len(quote_files), 0)

class TestFileSystemAdapter(unittest.TestCase):
    """文件系统适配器测试"""
    
    def setUp(self):
        import tempfile
        from src.infrastructure.storage.adapters.file_system import FileSystemAdapter
        self.temp_dir = tempfile.mkdtemp()
        self.adapter = FileSystemAdapter(self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    def test_write_read(self):
        """测试文件读写"""
        test_data = {"key": "value"}
        path = "test/path"

        # 写入测试
        write_result = self.adapter.write(path, test_data)
        self.assertTrue(write_result)

        # 读取验证
        read_data = self.adapter.read(path)
        self.assertEqual(read_data, test_data)

        # 验证文件路径
        expected_file = os.path.join(
            self.temp_dir,
            "test/path.json"
        )
        self.assertTrue(os.path.exists(expected_file))
