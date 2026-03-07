#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层文件系统工具组件测试

测试目标：提升utils/tools/file_system.py的真实覆盖率
实际导入和使用src.infrastructure.utils.tools.file_system模块
"""

import pytest
import tempfile
import os
import json
from pathlib import Path


class TestFileSystem:
    """测试文件系统工具类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.tools.file_system import FileSystem
        
        fs = FileSystem()
        assert fs is not None
    
    def test_create_directory(self):
        """测试创建目录"""
        from src.infrastructure.utils.tools.file_system import FileSystem
        
        fs = FileSystem()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test_dir")
            result = fs.create_directory(test_path)
            
            assert result is True
            assert os.path.exists(test_path)
            assert os.path.isdir(test_path)
    
    def test_create_directory_nested(self):
        """测试创建嵌套目录"""
        from src.infrastructure.utils.tools.file_system import FileSystem
        
        fs = FileSystem()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "level1", "level2", "level3")
            result = fs.create_directory(test_path)
            
            assert result is True
            assert os.path.exists(test_path)
    
    def test_create_directory_existing(self):
        """测试创建已存在的目录"""
        from src.infrastructure.utils.tools.file_system import FileSystem
        
        fs = FileSystem()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result1 = fs.create_directory(temp_dir)
            result2 = fs.create_directory(temp_dir)
            
            assert result1 is True
            assert result2 is True
    
    def test_list_directory(self):
        """测试列出目录内容"""
        from src.infrastructure.utils.tools.file_system import FileSystem
        
        fs = FileSystem()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试文件
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            
            files = fs.list_directory(temp_dir)
            
            assert isinstance(files, list)
            assert len(files) > 0
    
    def test_list_directory_empty(self):
        """测试列出空目录"""
        from src.infrastructure.utils.tools.file_system import FileSystem
        
        fs = FileSystem()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            files = fs.list_directory(temp_dir)
            
            assert isinstance(files, list)
    
    def test_list_directory_nonexistent(self):
        """测试列出不存在的目录"""
        from src.infrastructure.utils.tools.file_system import FileSystem
        
        fs = FileSystem()
        
        files = fs.list_directory("/nonexistent/path")
        
        assert isinstance(files, list)
        assert len(files) == 0
    
    def test_join_path(self):
        """测试连接路径"""
        from src.infrastructure.utils.tools.file_system import FileSystem
        
        fs = FileSystem()
        
        result = fs.join_path("path", "to", "file")
        
        assert isinstance(result, str)
        assert "path" in result


class TestFileSystemConstants:
    """测试文件系统常量类"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.tools.file_system import FileSystemConstants
        
        assert FileSystemConstants.DEFAULT_BASE_PATH == "data/storage"
        assert FileSystemConstants.JSON_FILE_SUFFIX == ".json"
        assert FileSystemConstants.JSON_INDENT_LEVEL == 2
        assert FileSystemConstants.DEFAULT_ENCODING == "utf-8"


class TestFileSystemAdapter:
    """测试文件系统适配器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FileSystemAdapter(base_path=temp_dir)
            
            assert adapter.base_path == Path(temp_dir)
            assert os.path.exists(temp_dir)
    
    def test_init_default_path(self):
        """测试使用默认路径初始化"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        adapter = FileSystemAdapter()
        
        assert adapter.base_path is not None
    
    def test_build_path(self):
        """测试构建路径"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FileSystemAdapter(base_path=temp_dir)
            path = adapter._build_path("test/data")
            
            assert path.suffix == ".json"
            assert "test" in str(path)
            assert "data" in str(path)
    
    def test_write(self):
        """测试写入数据"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FileSystemAdapter(base_path=temp_dir)
            data = {"key": "value", "number": 123}
            
            result = adapter.write("test/data", data)
            
            assert result is True
            assert os.path.exists(os.path.join(temp_dir, "test", "data.json"))
    
    def test_write_invalid_data(self):
        """测试写入无效数据"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FileSystemAdapter(base_path=temp_dir)
            
            # 尝试写入不可序列化的数据
            class Unserializable:
                pass
            
            result = adapter.write("test/data", {"obj": Unserializable()})
            
            assert result is False
    
    def test_read(self):
        """测试读取数据"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FileSystemAdapter(base_path=temp_dir)
            data = {"key": "value", "number": 123}
            
            adapter.write("test/data", data)
            result = adapter.read("test/data")
            
            assert result == data
    
    def test_read_nonexistent(self):
        """测试读取不存在的文件"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FileSystemAdapter(base_path=temp_dir)
            result = adapter.read("nonexistent/file")
            
            assert result is None
    
    def test_delete(self):
        """测试删除文件"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FileSystemAdapter(base_path=temp_dir)
            data = {"key": "value"}
            
            adapter.write("test/data", data)
            result = adapter.delete("test/data")
            
            assert result is True
            assert adapter.read("test/data") is None
    
    def test_delete_nonexistent(self):
        """测试删除不存在的文件"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FileSystemAdapter(base_path=temp_dir)
            result = adapter.delete("nonexistent/file")
            
            assert result is False
    
    def test_exists(self):
        """测试检查文件是否存在"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FileSystemAdapter(base_path=temp_dir)
            data = {"key": "value"}
            
            assert adapter.exists("test/data") is False
            
            adapter.write("test/data", data)
            assert adapter.exists("test/data") is True
    
    def test_save_load(self):
        """测试save和load方法（StorageAdapter接口兼容）"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FileSystemAdapter(base_path=temp_dir)
            data = {"key": "value"}
            
            result = adapter.save("test/key", data)
            assert result is True
            
            loaded = adapter.load("test/key")
            assert loaded == data
    
    def test_list_keys(self):
        """测试列出所有键"""
        from src.infrastructure.utils.tools.file_system import FileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FileSystemAdapter(base_path=temp_dir)
            adapter.write("test1/data1", {"key1": "value1"})
            adapter.write("test2/data2", {"key2": "value2"})
            
            keys = adapter.list_keys()
            
            assert isinstance(keys, list)
            assert len(keys) >= 2


class TestAShareFileSystemAdapter:
    """测试A股专用文件存储适配器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.tools.file_system import AShareFileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = AShareFileSystemAdapter(base_path=temp_dir)
            
            assert adapter.base_path == Path(temp_dir)
    
    def test_format_path(self):
        """测试格式化文件路径"""
        from src.infrastructure.utils.tools.file_system import AShareFileSystemAdapter
        
        adapter = AShareFileSystemAdapter()
        
        path = adapter.format_path("AAPL", "2024-01-01")
        
        assert path == "stock/AAPL/2024-01-01.parquet"
    
    def test_batch_write(self):
        """测试批量写入数据"""
        from src.infrastructure.utils.tools.file_system import AShareFileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = AShareFileSystemAdapter(base_path=temp_dir)
            batch_data = {
                "AAPL": {
                    "2024-01-01": {"price": 150.0},
                    "2024-01-02": {"price": 151.0}
                }
            }
            
            result = adapter.batch_write(batch_data)
            
            assert result is True
    
    def test_get_latest_data(self):
        """测试获取最新数据文件路径"""
        from src.infrastructure.utils.tools.file_system import AShareFileSystemAdapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = AShareFileSystemAdapter(base_path=temp_dir)
            
            # 创建测试文件
            test_dir = os.path.join(temp_dir, "stock", "AAPL")
            os.makedirs(test_dir, exist_ok=True)
            test_file1 = os.path.join(test_dir, "2024-01-01.parquet")
            test_file2 = os.path.join(test_dir, "2024-01-02.parquet")
            with open(test_file1, 'w') as f:
                f.write("test1")
            with open(test_file2, 'w') as f:
                f.write("test2")
            
            latest = adapter.get_latest_data("AAPL")
            
            assert latest is not None
            assert "2024-01-02" in latest

