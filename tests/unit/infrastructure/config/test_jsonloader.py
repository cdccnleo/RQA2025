import json
import time
import pathlib
import os

import pytest
from pathlib import Path
from src.infrastructure.config.strategies.json_loader import JSONLoader
from src.infrastructure.config.exceptions import ConfigLoadError, ConfigValidationError
from unittest.mock import MagicMock, patch, mock_open

class MockPath:
    def __init__(self, path):
        self.path = str(path)
        self.exists_value = True
        self.size_value = 0
        
    def exists(self):
        return self.exists_value
        
    def stat(self):
        # 确保返回正确的文件大小
        class StatResult:
            def __init__(self, size):
                self.st_size = size
        return StatResult(self.size_value)
        
    def absolute(self):
        return self
        
    def __str__(self):
        return self.path
        
    def __fspath__(self):
        return self.path

# 全局mock_path实例
mock_path = None

def mock_path_init(cls, path):
    # 使用全局mock_path实例
    global mock_path
    mock = MockPath(path)
    # 确保保留所有mock属性
    if mock_path is not None:
        if hasattr(mock_path, 'exists_value'):
            mock.exists_value = mock_path.exists_value
        if hasattr(mock_path, 'size_value'):
            mock.size_value = mock_path.size_value
            # 确保stat()方法返回正确的文件大小
            mock.stat = lambda: type('', (), {'st_size': mock.size_value})()
    return mock

def mock_path_absolute(self):
    return self


@pytest.fixture
def json_loader():
    """Fixture that provides a JSONLoader instance"""
    return JSONLoader()

class TestJSONLoader:
    @pytest.mark.unit
    def test_load_valid_json(self, json_loader, tmp_path):
        """Test loading a valid JSON file"""
        test_data = {"database": {"host": "localhost"}}
        json_str = json.dumps(test_data)
        test_file = tmp_path / "valid.json"
        test_file.write_text(json_str)

        config, meta = json_loader.load(str(test_file))

        assert config == test_data
        assert meta["format"] == "json"
        assert meta["size"] == len(json_str)
        assert meta["load_time"] > 0

    @pytest.mark.unit
    def test_load_invalid_json(self, json_loader):
        """测试加载无效JSON文件"""
        invalid_json = "{invalid}"
        mock_path = MockPath("invalid.json")
        mock_path.exists_value = True
        mock_path.size_value = len(invalid_json)

        with patch("builtins.open", mock_open(read_data=invalid_json)), \
             patch("pathlib.Path.__new__", side_effect=mock_path_init), \
             patch("pathlib.Path.absolute", mock_path_absolute):
            with pytest.raises(ConfigValidationError, match="JSON解析失败"):
                json_loader.load("invalid.json")

    @pytest.mark.unit
    def test_file_not_found(self, json_loader):
        """测试文件不存在情况"""
        mock_path = MockPath("nonexistent.json")
        mock_path.exists_value = False

        with patch("pathlib.Path.__new__", side_effect=mock_path_init), \
             patch("pathlib.Path.absolute", mock_path_absolute):
            with pytest.raises(ConfigLoadError):
                json_loader.load("nonexistent.json")

    @pytest.mark.unit
    def test_batch_load(self, json_loader):
        """测试批量加载JSON文件"""
        files = ["config1.json", "config2.json"]
        test_data = {"key": "value"}
        json_str = json.dumps(test_data)

        # 为每个文件创建单独的MockPath实例
        mock_paths = []
        for file in files:
            mock_path = MockPath(file)
            mock_path.exists_value = True
            mock_path.size_value = len(json_str)
            mock_paths.append(mock_path)

        # 修改mock_path_init以返回对应的MockPath实例
        def custom_mock_path_init(self, path):
            for mock in mock_paths:
                if mock.path == str(path):
                    return mock
            return MockPath(path)

        with patch("builtins.open", mock_open(read_data=json_str)), \
             patch("pathlib.Path.__new__", side_effect=custom_mock_path_init), \
             patch("pathlib.Path.absolute", mock_path_absolute):
            results = json_loader.batch_load(files)
            assert len(results) == 2

            assert len(results) == 2
            for file in files:
                assert file in results
                assert results[file][0] == test_data
                assert results[file][1]["size"] == len(json_str)

    @pytest.mark.performance
    def test_large_file_loading(self, json_loader):
        """测试大文件加载性能"""
        # 生成大JSON数据 (约1MB)
        large_data = {"key": "x" * 1024 * 1024}
        json_str = json.dumps(large_data)

        # 设置全局mock_path实例
        global mock_path
        mock_path = MockPath("large.json")
        mock_path.exists_value = True
        mock_path.size_value = len(json_str)

        with patch("builtins.open", mock_open(read_data=json_str)), \
             patch("pathlib.Path.__new__", side_effect=mock_path_init), \
             patch("pathlib.Path.absolute", mock_path_absolute):
            start = time.time()
            config, meta = json_loader.load("large.json")
            end = time.time()

            assert meta["size"] == len(json_str)
            assert (end - start) < 1.0  # 1MB文件应在1秒内加载
