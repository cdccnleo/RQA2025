#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_utils 模块补充测试

聚焦安全写入/读取及异常分支，提升工具模块覆盖率。
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any

import pytest

from src.infrastructure.utils.tools import file_utils
from src.infrastructure.utils.tools.file_utils import (
    FileUtils,
    delete_file,
    ensure_directory,
    get_file_size,
    list_files,
    safe_file_read,
    safe_file_write,
)


class _DummyObject:
    """用于触发 pickle 序列化路径的辅助类."""

    def __init__(self, value: Any):
        self.value = value


def test_file_utils_basic_operations(tmp_path: Path) -> None:
    """验证 FileUtils 的基础读写/复制/删除流程。"""
    utils = FileUtils()
    source = tmp_path / "source.txt"
    copy_target = tmp_path / "copy.txt"

    assert utils.write_file(str(source), "hello")
    assert utils.read_file(str(source)) == "hello"
    assert utils.copy_file(str(source), str(copy_target))
    assert copy_target.read_text(encoding="utf-8") == "hello"

    # 缺失文件读取返回空字符串
    missing = tmp_path / "missing.txt"
    assert utils.read_file(str(missing)) == ""

    assert utils.delete_file(str(copy_target)) is True
    assert not copy_target.exists()

    # 删除不存在文件返回 False
    assert utils.delete_file(str(copy_target)) is False


def test_file_utils_write_file_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """write_file 出现异常时应返回 False。"""
    utils = FileUtils()

    def _boom(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(file_utils, "open", _boom, raising=False)
    assert utils.write_file(str(tmp_path / "fail.txt"), "content") is False


def test_safe_file_write_json_and_bytes(tmp_path: Path) -> None:
    """safe_file_write 支持 dict 和 bytes 内容。"""
    json_path = tmp_path / "data.json"
    bytes_path = tmp_path / "data.bin"

    assert safe_file_write(json_path, {"value": 42})
    with json_path.open("r", encoding="utf-8") as fh:
        assert json.load(fh) == {"value": 42}

    payload = b"binary-data"
    assert safe_file_write(bytes_path, payload)
    with bytes_path.open("rb") as fh:
        assert fh.read() == payload


def test_safe_file_write_pickle_fallback(tmp_path: Path) -> None:
    """非基础类型应走 pickle 序列化分支。"""
    target = tmp_path / "object.pkl"
    dummy = _DummyObject({"k": "v"})

    assert safe_file_write(target, dummy)
    with target.open("rb") as fh:
        restored = pickle.load(fh)
    assert isinstance(restored, _DummyObject)
    assert restored.value == {"k": "v"}


def test_safe_file_write_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """写入异常时应返回 False 并吞掉异常。"""

    def _boom(*args, **kwargs):
        raise OSError("no permission")

    monkeypatch.setattr(file_utils, "open", _boom, raising=False)
    assert safe_file_write(tmp_path / "should_fail.txt", "content") is False


def test_safe_file_read_json_text_binary(tmp_path: Path) -> None:
    """覆盖 safe_file_read 的多种解析路径。"""
    json_file = tmp_path / "sample.json"
    text_file = tmp_path / "sample.txt"
    pickle_file = tmp_path / "sample.pkl"

    json_file.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    text_file.write_text("plain text", encoding="utf-8")
    with pickle_file.open("wb") as fh:
        pickle.dump({"binary": True}, fh)

    assert safe_file_read(json_file) == {"foo": "bar"}
    assert safe_file_read(text_file) == "plain text"
    assert safe_file_read(pickle_file, encoding="utf-8") == {"binary": True}


def test_safe_file_read_missing_and_type_error() -> None:
    """缺失文件与无效入参均应返回 None。"""
    assert safe_file_read(Path("non_exists_file.tmp")) is None
    assert safe_file_read(None) is None  # type: ignore[arg-type]


def test_ensure_directory_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ensure_directory 捕获目录创建异常。"""

    def _boom(self, parents=True, exist_ok=True):
        raise OSError("denied")

    monkeypatch.setattr(file_utils.Path, "mkdir", _boom)
    assert ensure_directory(tmp_path / "subdir") is False


def test_list_files_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """list_files 在 glob 失败时返回空列表。"""

    def _boom(self, pattern="*"):
        raise OSError("fail glob")

    monkeypatch.setattr(file_utils.Path, "glob", _boom)
    assert list_files(tmp_path, "*.txt") == []


def test_get_file_size_success_and_failure(tmp_path: Path) -> None:
    """覆盖 get_file_size 的成功与失败路径。"""
    existing = tmp_path / "file.txt"
    existing.write_text("abc", encoding="utf-8")
    assert get_file_size(existing) == len("abc")

    assert get_file_size(tmp_path / "missing.txt") == 0


def test_delete_file_success_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """delete_file 删除成功与异常分支。"""
    target = tmp_path / "temp.txt"
    target.write_text("delete me", encoding="utf-8")
    assert delete_file(target) is True
    assert delete_file(target) is True  # missing_ok=True 覆盖第二次删除

    def _boom(self, missing_ok=True):
        raise OSError("cannot remove")

    monkeypatch.setattr(file_utils.Path, "unlink", _boom)
    assert delete_file(tmp_path / "other.txt") is False

