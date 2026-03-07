#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JSON配置加载器测试"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from src.infrastructure.config.loaders.json_loader import JSONLoader
from src.infrastructure.config.config_exceptions import ConfigLoadError


class TestJSONLoader:
    """测试JSON配置加载器"""

    def setup_method(self):
        """测试前准备"""
        self.loader = JSONLoader()

    def test_init(self):
        """测试初始化"""
        assert self.loader.name == "JSONLoader"
        assert self.loader.format.value == "json"
        assert isinstance(self.loader._last_metadata, dict)

    def test_can_load_json_file(self):
        """测试可以加载JSON文件"""
        assert self.loader.can_load("config.json") is True
        assert self.loader.can_load("settings.JSON") is True
        assert self.loader.can_load("/path/to/config.json") is True

    def test_can_load_non_json_file(self):
        """测试不能加载非JSON文件"""
        assert self.loader.can_load("config.yaml") is False
        assert self.loader.can_load("config.toml") is False
        assert self.loader.can_load("config.txt") is False
        assert self.loader.can_load("") is False
        assert self.loader.can_load(None) is False

    def test_can_handle_source(self):
        """测试处理源能力"""
        # can_handle_source应该与can_load行为一致
        assert self.loader.can_handle_source("config.json") is True
        assert self.loader.can_handle_source("config.yaml") is False

    def test_get_supported_formats(self):
        """测试支持的格式"""
        formats = self.loader.get_supported_formats()
        assert len(formats) == 1
        assert formats[0].value == "json"

    def test_get_supported_extensions(self):
        """测试支持的扩展名"""
        extensions = self.loader.get_supported_extensions()
        assert extensions == [".json"]

    def test_get_last_metadata_empty(self):
        """测试获取空的最后元数据"""
        metadata = self.loader.get_last_metadata()
        assert isinstance(metadata, dict)
        assert len(metadata) == 0

    def test_load_valid_json_file(self):
        """测试加载有效的JSON文件"""
        config_data = {"database": {"host": "localhost", "port": 5432}, "cache": {"enabled": True}}

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name

        try:
            result = self.loader.load(temp_file)

            # 对于字典数据，应该返回LoaderResult对象（继承自dict）
            assert isinstance(result, dict)
            assert hasattr(result, 'metadata')
            assert dict(result) == config_data

            # 验证元数据
            assert result.metadata['format'] == 'json'
            assert result.metadata['source'] == temp_file
            assert 'timestamp' in result.metadata
            assert 'load_time' in result.metadata
            assert result.metadata['size'] > 0

            # 验证最后元数据
            last_metadata = self.loader.get_last_metadata()
            assert last_metadata['format'] == 'json'

        finally:
            os.unlink(temp_file)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(ConfigLoadError) as exc_info:
            self.loader.load("/nonexistent/path/config.json")

        error_msg = str(exc_info.value)
        assert "JSON文件不存在" in error_msg
        assert "file_not_found" in error_msg

    def test_load_invalid_json_file(self):
        """测试加载无效的JSON文件"""
        # 创建包含无效JSON的临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json syntax}')
            temp_file = f.name

        try:
            with pytest.raises(ConfigLoadError) as exc_info:
                self.loader.load(temp_file)

            error_msg = str(exc_info.value)
            assert "JSON解析失败" in error_msg

        finally:
            os.unlink(temp_file)

    def test_load_empty_json_file(self):
        """测试加载空JSON文件"""
        # 创建空JSON文件的临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{}')
            temp_file = f.name

        try:
            result = self.loader.load(temp_file)

            # 空字典也应该返回LoaderResult
            assert isinstance(result, dict)
            assert dict(result) == {}

        finally:
            os.unlink(temp_file)

    def test_batch_load_valid_files(self):
        """测试批量加载有效文件"""
        config1 = {"service": "api", "port": 8080}
        config2 = {"service": "db", "host": "localhost"}

        # 创建临时文件
        temp_files = []
        try:
            # 创建第一个文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config1, f)
                temp_files.append(f.name)

            # 创建第二个文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config2, f)
                temp_files.append(f.name)

            result = self.loader.batch_load(temp_files)

            assert len(result) == 2
            assert temp_files[0] in result
            assert temp_files[1] in result

            # 验证数据
            data1, metadata1 = result[temp_files[0]]
            data2, metadata2 = result[temp_files[1]]

            assert data1 == config1
            assert data2 == config2

            # 验证元数据
            assert metadata1['format'] == 'json'
            assert metadata2['format'] == 'json'

        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_batch_load_with_invalid_file(self):
        """测试批量加载包含无效文件的情况"""
        config_data = {"valid": "data"}

        # 创建临时文件
        temp_files = []
        try:
            # 创建有效文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                temp_files.append(f.name)

            # 创建无效文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write('invalid json content')
                temp_files.append(f.name)

            # 应该抛出异常
            with pytest.raises(ConfigLoadError) as exc_info:
                self.loader.batch_load(temp_files)

            error_msg = str(exc_info.value)
            assert "批量加载失败" in error_msg

        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_batch_load_empty_list(self):
        """测试批量加载空列表"""
        result = self.loader.batch_load([])
        assert result == {}

    def test_load_non_dict_json(self):
        """测试加载非字典JSON数据"""
        test_cases = [
            [1, 2, 3],  # 数组
            "string",   # 字符串
            42,         # 数字
            True,       # 布尔值
            None        # null
        ]

        for test_data in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(test_data, f)
                temp_file = f.name

            try:
                result = self.loader.load(temp_file)

                # 对于非字典数据，应该直接返回数据
                assert result == test_data

            finally:
                os.unlink(temp_file)

    def test_load_file_with_unicode_content(self):
        """测试加载包含Unicode内容的JSON文件"""
        config_data = {
            "message": "你好世界",
            "emoji": "🚀📊✅",
            "special_chars": "äöüñ"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8', delete=False) as f:
            json.dump(config_data, f, ensure_ascii=False)
            temp_file = f.name

        try:
            result = self.loader.load(temp_file)

            # Unicode字典应该返回LoaderResult
            assert isinstance(result, dict)
            assert dict(result) == config_data
            assert result["message"] == "你好世界"
            assert result["emoji"] == "🚀📊✅"

        finally:
            os.unlink(temp_file)
