#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON Loader 测试

测试 src/infrastructure/config/loaders/json_loader.py 文件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import patch, mock_open

# 尝试导入模块
try:
    from src.infrastructure.config.loaders.json_loader import JSONLoader
    from src.infrastructure.config.config_exceptions import ConfigLoadError
    from src.infrastructure.config.interfaces.unified_interface import ConfigFormat
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestJSONLoader:
    """测试JSONLoader类"""

    def setup_method(self):
        """测试前准备"""
        self.loader = JSONLoader()

    def test_initialization(self):
        """测试初始化"""
        loader = JSONLoader()
        assert loader is not None

    def test_load_valid_json_file(self):
        """测试加载有效JSON文件"""
        # 创建临时JSON文件
        test_data = {"key1": "value1", "key2": 42, "nested": {"inner": "value"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            temp_file = f.name
        
        try:
            result = self.loader.load(temp_file)
            assert result == test_data
        finally:
            os.unlink(temp_file)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(ConfigLoadError) as exc_info:
            self.loader.load("nonexistent.json")
        
        error_msg = str(exc_info.value)
        assert "JSON文件不存在" in error_msg
        assert "nonexistent.json" in error_msg
        assert "file_not_found" in error_msg

    def test_load_invalid_json_file(self):
        """测试加载无效JSON文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            f.write("{ invalid json content")
            temp_file = f.name
        
        try:
            with pytest.raises(ConfigLoadError) as exc_info:
                self.loader.load(temp_file)
            
            error_msg = str(exc_info.value)
            assert "JSON解析失败" in error_msg
            assert temp_file in error_msg
        finally:
            os.unlink(temp_file)

    def test_load_file_with_io_error(self):
        """测试文件IO错误"""
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(ConfigLoadError) as exc_info:
                self.loader.load("test.json")
            
            error_msg = str(exc_info.value)
            assert "加载JSON文件失败" in error_msg
            assert "Permission denied" in error_msg

    def test_batch_load_valid_files(self):
        """测试批量加载有效文件"""
        test_files = []
        test_data = [
            {"file1": "data1"},
            {"file2": "data2", "number": 123}
        ]
        
        try:
            # 创建临时文件
            for i, data in enumerate(test_data):
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{i}.json', delete=False, encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                    test_files.append(f.name)
            
            results = self.loader.batch_load(test_files)
            
            assert len(results) == len(test_files)
            for i, file_path in enumerate(test_files):
                assert file_path in results
                data, meta = results[file_path]
                assert data == test_data[i]
                assert meta['format'] == 'json'
                assert meta['source'] == file_path
                assert 'timestamp' in meta
                assert meta['size'] > 0
        finally:
            for file_path in test_files:
                if os.path.exists(file_path):
                    os.unlink(file_path)

    def test_batch_load_with_invalid_file(self):
        """测试批量加载包含无效文件"""
        valid_data = {"valid": "data"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(valid_data, f)
            valid_file = f.name
        
        try:
            with pytest.raises(ConfigLoadError) as exc_info:
                self.loader.batch_load([valid_file, "nonexistent.json"])
            
            error_msg = str(exc_info.value)
            assert "批量加载失败" in error_msg
            assert "nonexistent.json" in error_msg
        finally:
            os.unlink(valid_file)

    def test_batch_load_empty_list(self):
        """测试批量加载空列表"""
        results = self.loader.batch_load([])
        assert results == {}

    def test_can_load_valid_extension(self):
        """测试can_load有效扩展名"""
        assert self.loader.can_load("config.json") is True
        assert self.loader.can_load("test.JSON") is True
        assert self.loader.can_load("/path/to/file.json") is True

    def test_can_load_invalid_extension(self):
        """测试can_load无效扩展名"""
        assert self.loader.can_load("config.yaml") is False
        assert self.loader.can_load("config.txt") is False
        assert self.loader.can_load("config") is False
        assert self.loader.can_load("") is False

    def test_can_load_non_string(self):
        """测试can_load非字符串输入"""
        assert self.loader.can_load(123) is False
        assert self.loader.can_load(None) is False
        assert self.loader.can_load([]) is False

    def test_can_handle_source(self):
        """测试can_handle_source方法"""
        # 应该与can_load行为相同
        assert self.loader.can_handle_source("test.json") is True
        assert self.loader.can_handle_source("test.yaml") is False

    def test_get_supported_formats(self):
        """测试get_supported_formats方法"""
        formats = self.loader.get_supported_formats()
        assert ConfigFormat.JSON in formats
        assert len(formats) == 1

    def test_get_supported_extensions(self):
        """测试get_supported_extensions方法"""
        extensions = self.loader.get_supported_extensions()
        assert '.json' in extensions
        assert len(extensions) == 1

    def test_load_complex_json(self):
        """测试加载复杂JSON数据"""
        complex_data = {
            "array": [1, 2, {"nested": True}],
            "string": "test with unicode: 测试",
            "number": 3.14159,
            "boolean": False,
            "null_value": None
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(complex_data, f, ensure_ascii=False)
            temp_file = f.name
        
        try:
            result = self.loader.load(temp_file)
            assert result == complex_data
            assert result["string"] == "test with unicode: 测试"
            assert result["array"][2]["nested"] is True
        finally:
            os.unlink(temp_file)

    def test_load_empty_json_object(self):
        """测试加载空JSON对象"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump({}, f)
            temp_file = f.name
        
        try:
            result = self.loader.load(temp_file)
            assert result == {}
        finally:
            os.unlink(temp_file)

    def test_load_json_array(self):
        """测试加载JSON数组"""
        array_data = [1, "test", {"key": "value"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(array_data, f)
            temp_file = f.name
        
        try:
            result = self.loader.load(temp_file)
            assert result == array_data
        finally:
            os.unlink(temp_file)


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestJSONLoaderIntegration:
    """测试JSONLoader集成功能"""

    def setup_method(self):
        """测试前准备"""
        self.loader = JSONLoader()

    def test_error_details_in_exception(self):
        """测试异常中的详细信息"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json, "missing": quotes}')
            temp_file = f.name
        
        try:
            with pytest.raises(ConfigLoadError) as exc_info:
                JSONLoader().load(temp_file)
            
            exception = exc_info.value
            assert hasattr(exception, 'config_key')
            # 验证错误消息包含有用的信息
            error_msg = str(exception)
            assert temp_file in error_msg
        finally:
            os.unlink(temp_file)

    def test_batch_load_error_details(self):
        """测试批量加载错误详情"""
        valid_file = None
        try:
            # 创建有效文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({"valid": "data"}, f)
                valid_file = f.name
            
            with pytest.raises(ConfigLoadError) as exc_info:
                self.loader.batch_load([valid_file, "nonexistent.json"])
            
            exception = exc_info.value
            assert hasattr(exception, 'details')
            assert exception.details is not None
        finally:
            if valid_file and os.path.exists(valid_file):
                os.unlink(valid_file)

    def test_path_handling(self):
        """测试路径处理"""
        # 测试相对路径和绝对路径
        test_data = {"test": "data"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            # 测试绝对路径
            abs_path = os.path.abspath(temp_file)
            result1 = self.loader.load(abs_path)
            assert result1 == test_data
            
            # 测试相对路径（如果可能）
            rel_path = os.path.relpath(temp_file)
            if rel_path != abs_path:
                result2 = self.loader.load(rel_path)
                assert result2 == test_data
        finally:
            os.unlink(temp_file)

    def test_metadata_generation(self):
        """测试元数据生成"""
        test_data = {"meta": "test"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            results = self.loader.batch_load([temp_file])
            assert temp_file in results
            
            data, meta = results[temp_file]
            assert data == test_data
            
            # 验证元数据字段
            assert meta['format'] == 'json'
            assert meta['source'] == temp_file
            assert isinstance(meta['timestamp'], (int, float))
            assert meta['timestamp'] > 0
            assert isinstance(meta['size'], int)
            assert meta['size'] > 0
        finally:
            os.unlink(temp_file)

    def test_unicode_handling(self):
        """测试Unicode处理"""
        unicode_data = {
            "chinese": "中文测试",
            "emoji": "🚀测试",
            "special": "特殊字符: ąćęłńóśźż"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(unicode_data, f, ensure_ascii=False)
            temp_file = f.name
        
        try:
            result = self.loader.load(temp_file)
            assert result == unicode_data
            assert result["chinese"] == "中文测试"
            assert result["emoji"] == "🚀测试"
        finally:
            os.unlink(temp_file)
