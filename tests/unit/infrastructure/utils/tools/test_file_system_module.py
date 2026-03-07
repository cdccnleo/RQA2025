#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_system 模块测试，覆盖 FileSystem/Adapter 核心功能。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.utils.tools.file_system import (
    AShareFileSystemAdapter,
    FileSystem,
    FileSystemAdapter,
)


def test_file_system_create_and_list(tmp_path: Path, monkeypatch):
    fs = FileSystem()
    assert fs.create_directory(tmp_path / "dir")
    (tmp_path / "dir" / "file.txt").write_text("data", encoding="utf-8")
    entries = fs.list_directory(tmp_path / "dir")
    assert len(entries) == 1
    assert "file.txt" in entries[0]
    assert fs.join_path(tmp_path, "dir", "file.txt").endswith("file.txt")

    def boom(*args, **kwargs):
        raise OSError("fail")

    monkeypatch.setattr(Path, "mkdir", boom, raising=False)
    assert fs.create_directory(tmp_path / "bad") is False


def test_file_system_adapter_write_read_delete(tmp_path: Path):
    adapter = FileSystemAdapter(base_path=str(tmp_path))
    data = {"name": "test"}
    assert adapter.write("sample/path", data) is True

    read_data = adapter.read("sample/path")
    assert read_data == data
    assert adapter.exists("sample/path") is True
    keys = adapter.list_keys()
    assert "sample/path" in keys

    assert adapter.delete("sample/path") is True
    assert adapter.read("sample/path") is None
    assert adapter.delete("missing") is False


def test_file_system_adapter_write_failure(monkeypatch, tmp_path: Path):
    adapter = FileSystemAdapter(base_path=str(tmp_path))

    def boom(*args, **kwargs):
        raise IOError("cannot write")

    monkeypatch.setattr("builtins.open", boom)
    assert adapter.write("sample", {"x": 1}) is False


def test_file_system_adapter_read_invalid_json(tmp_path: Path):
    adapter = FileSystemAdapter(base_path=str(tmp_path))
    target = adapter._build_path("bad")  # type: ignore[attr-defined]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{bad json]", encoding="utf-8")
    assert adapter.read("bad") is None


def test_a_share_adapter_batch_and_latest(tmp_path: Path):
    adapter = AShareFileSystemAdapter(base_path=str(tmp_path))
    batch = {
        "AAA": {
            "20250101": {"close": 1},
            "20250102": {"close": 2},
        }
    }
    assert adapter.batch_write(batch) is True
    latest = adapter.get_latest_data("AAA")
    assert latest is not None
    assert latest.endswith("20250102.parquet")

    assert adapter.get_latest_data("BBB") is None


def test_a_share_adapter_batch_failure(monkeypatch, tmp_path: Path):
    adapter = AShareFileSystemAdapter(base_path=str(tmp_path))

    def fail_write(self, path, data):
        return False

    monkeypatch.setattr(FileSystemAdapter, "write", fail_write)
    assert adapter.batch_write({"AAA": {"20250101": {"close": 1}}}) is False

