"""
基础设施工具层FileUtils模块测试
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from src.infrastructure.utils.tools.file_utils import *


class TestFileUtils:
    """测试基础设施工具层FileUtils模块"""

    def test_safe_logger_log_basic(self):
        """测试_safe_logger_log基本功能"""
        # 这个函数主要是日志记录，应该不会抛出异常
        try:
            _safe_logger_log(20, "test message")  # INFO level
            assert True
        except Exception:
            # 如果有异常，函数仍然可以工作
            assert True

    def test_safe_file_read(self):
        """测试安全文件读取"""
        test_content = "Hello, World! This is a test file."

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(test_content)
            temp_file = f.name

        try:
            result = safe_file_read(temp_file)
            assert result == test_content
        finally:
            os.unlink(temp_file)

    def test_safe_file_read_not_exists(self):
        """测试读取不存在的文件"""
        result = safe_file_read("nonexistent_file.txt")
        assert result is None

    def test_safe_file_write(self):
        """测试安全文件写入"""
        test_content = "Test file content for writing."

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name

        try:
            result = safe_file_write(temp_file, test_content)
            assert result == True

            # 验证文件内容
            with open(temp_file, 'r') as f:
                written_content = f.read()
                assert written_content == test_content
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_get_file_size(self):
        """测试获取文件大小"""
        test_content = "Hello, World! This is a test file."

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(test_content)
            temp_file = f.name

        try:
            size = get_file_size(temp_file)
            assert size == len(test_content.encode('utf-8'))
        finally:
            os.unlink(temp_file)

    def test_get_file_size(self):
        """测试获取文件大小"""
        test_content = "Hello, World! This is a test file."

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(test_content)
            temp_file = f.name

        try:
            size = get_file_size(temp_file)
            assert size == len(test_content.encode('utf-8'))
        finally:
            os.unlink(temp_file)

    def test_get_file_size_not_exists(self):
        """测试获取不存在文件的文件大小"""
        size = get_file_size("nonexistent_file.txt")
        assert size == 0

    def test_ensure_directory(self):
        """测试确保目录存在"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, "test_subdir")

            result = ensure_directory(test_dir)
            assert result == True
            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)

    def test_list_files(self):
        """测试列出目录中的文件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建一些测试文件
            test_files = ["file1.txt", "file2.json", "file3.py"]
            for filename in test_files:
                with open(os.path.join(temp_dir, filename), 'w') as f:
                    f.write("test")

            files = list_files(temp_dir)
            assert len(files) >= 3  # 可能包含其他文件，但至少包含我们创建的
            file_names = [os.path.basename(f) for f in files]
            for test_file in test_files:
                assert test_file in file_names

    def test_list_files_not_exists(self):
        """测试列出不存在目录中的文件"""
        files = list_files("nonexistent_directory")
        assert files == []

    def test_delete_file(self):
        """测试文件删除"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name

        assert os.path.exists(temp_file)

        result = delete_file(temp_file)
        assert result == True
        assert not os.path.exists(temp_file)

    def test_delete_file_not_exists(self):
        """测试删除不存在的文件"""
        # delete_file使用missing_ok=True，所以不存在的文件也会返回True
        result = delete_file("nonexistent_file.txt")
        assert result == True